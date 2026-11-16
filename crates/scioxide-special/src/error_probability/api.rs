//! Family-owned plain, detailed, and scoped scalar entry points.
use crate::SpecialFunctionResult;

use scioxide_core::{Complex, NumericScalar};
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the error function erf(z).

```text
erf(z) = (2/√π) ∫₀ᶻ exp(−t²) dt
```

(DLMF 7.2.1). erf is entire and odd, with `erf(0) = 0` and `erf(x) → ±1` as `x → ±∞`. It
accepts real and complex arguments in `f32` or `f64`.

Real arguments use fixed-cost approximations: a rational approximation for `|x| ≤ 1` and
`1 − exp(−x²) erfcx(x)` beyond. Complex arguments are evaluated through the Faddeeva function
[`faddeeva`](crate::faddeeva).

# Examples

```
// erf(1) = 0.8427007929497149
let e = scioxide_special::erf(1.0_f64)?;
assert!((e - 0.842_700_792_949_714_9).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
Real arguments are evaluated in `f64` and rounded once for `f32`. Complex arguments have
no uniform ulp target; the detailed form reports an error estimate. See the [crate
accuracy table](crate#accuracy)."###, r###"
Real arguments have no error conditions of their own. For complex `z`:

- `NumericalInstability` (`Overflow`) if a complex value exceeds the range of the scalar type.
- `NumericalInstability` (`LossOfSignificance`) if `Re z · Im z` overflows, so the phase of
  `exp(−z²)` cannot be formed.

# References

- NIST Digital Library of Mathematical Functions, §7.2. <https://dlmf.nist.gov/7.2>
- M. R. Zaghloul and A. N. Ali, "Algorithm 916: Computing the Faddeyeva and Voigt functions", *ACM
  Trans. Math. Softw.* 38(2), 2011."###)]
    #[inline]
    pub erf [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::ErrorFunctionOptions;
    detailed erf_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::error_probability::error_fn::erf(x.real_component()).map(T::from_real)
    } complex {
        crate::error_probability::complex_error::erf_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::error_probability::real_api::erf_evaluation(x.real_component(), options, context)
    } complex {
        crate::error_probability::complex_error::erf_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    complex scoped pub erf_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complementary error function erfc(z) = 1 − erf(z).

```text
erfc(z) = 1 − erf(z) = (2/√π) ∫_z^∞ exp(−t²) dt
```

(DLMF 7.2.2). erfc is entire, with `erfc(0) = 1`, `erfc(−z) = 2 − erfc(z)` and `erfc(x) → 0`
as `x → +∞`. It accepts real and complex arguments in `f32` or `f64`.

The real tail is computed directly as `exp(−x²) erfcx(x)`, with `x²` formed exactly, so the
result keeps full relative accuracy instead of losing it to the subtraction `1 − erf(x)`. For
`x > 27.3` the value is below the smallest `f64` subnormal and the result is zero; use
[`erfcx`](crate::erfcx) or [`log_erfc`](crate::log_erfc) there. Complex arguments are evaluated
through the Faddeeva function [`faddeeva`](crate::faddeeva).

# Examples

```
// erfc(1) = 1 − erf(1) = 0.1572992070502851
let e = scioxide_special::erfc(1.0_f64)?;
assert!((e - 0.157_299_207_050_285_1).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
Real arguments are evaluated in `f64` and rounded once for `f32`. Complex arguments have
no uniform ulp target; the detailed form reports an error estimate. See the [crate
accuracy table](crate#accuracy)."###, r###"
Real arguments have no error conditions of their own. For complex `z`:

- `NumericalInstability` (`Overflow`) if a complex value exceeds the range of the scalar type.
- `NumericalInstability` (`LossOfSignificance`) if `Re z · Im z` overflows, so the phase of
  `exp(−z²)` cannot be formed.

# References

- NIST Digital Library of Mathematical Functions, §7.2. <https://dlmf.nist.gov/7.2>"###)]
    #[inline]
    pub erfc [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::ErrorFunctionOptions;
    detailed erfc_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::error_probability::error_fn::erfc(x.real_component()).map(T::from_real)
    } complex {
        crate::error_probability::complex_error::erfc_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::error_probability::real_api::erfc_evaluation(x.real_component(), options, context)
    } complex {
        crate::error_probability::complex_error::erfc_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    complex scoped pub erfc_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled complementary error function erfcx(z) = exp(z²) erfc(z).

```text
erfcx(z) = exp(z²) erfc(z) = w(iz)
```

where `w` is the Faddeeva function (DLMF 7.2.3). erfcx is entire. For large positive `x`,
`erfcx(x) ≈ 1/(x√π)`, so it stays representable where `erfc(x)` underflows. For negative `x`
it grows like `2 exp(x²)` and overflows below about `x = −26.64` in `f64` and `x = −9.38` in
`f32`.

Real arguments use piecewise degree-9 Chebyshev interpolants on `[−1, 32)`, sixteen per binade
above 1, an asymptotic expansion in `1/x²` beyond 32, and the reflection
`erfcx(x) = 2 exp(x²) − erfcx(−x)` below −1. The cost is fixed. Complex arguments are evaluated
through [`faddeeva`](crate::faddeeva).

# Examples

```
// erfcx(x) = exp(x²) erfc(x)
let x = 1.5_f64;
let expected = (x * x).exp() * scioxide_special::erfc(x)?;
assert!((scioxide_special::erfcx(x)? - expected).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested
range. The detailed form reports a bound of 2 units for `f64` and 1 unit for `f32`
([error estimates](crate#error-estimates)). Complex arguments have no uniform ulp target;
the detailed form reports an error estimate. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if a real `x` is below about −26.64 (`f64`) or −9.38 (`f32`).
  [`erfcx_with_options`](crate::erfcx_with_options) still returns the value in scaled and
  logarithmic form.
- `NumericalInstability` (`Overflow`) if a complex value exceeds the range of the scalar type.
- `NumericalInstability` (`LossOfSignificance`) if `Re z · Im z` overflows, so the phase of
  `exp(−z²)` cannot be formed.

# References

- NIST Digital Library of Mathematical Functions, §7.2. <https://dlmf.nist.gov/7.2>"###)]
    #[inline]
    pub erfcx [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::ErrorFunctionOptions;
    detailed erfcx_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::error_probability::probability::erfcx(x.real_component()).map(T::from_real)
    } complex {
        crate::error_probability::complex_error::erfcx_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::error_probability::probability::erfcx_evaluation(x.real_component(), options, context)
    } complex {
        crate::error_probability::complex_error::erfcx_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    complex scoped pub erfcx_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the imaginary error function erfi(z) = −i erf(iz).

```text
erfi(z) = −i erf(iz) = (2/√π) ∫₀ᶻ exp(t²) dt
```

erfi is entire and odd. For real `x` it equals `(2/√π) exp(x²) D(x)`, where `D` is Dawson's
integral ([`dawson`](crate::dawson)), and grows like `exp(x²)/(x√π)`.

Real arguments use this relation to Dawson's integral, at fixed cost. The factor `exp(x²)` is
kept as a separate scale, so the detailed form returns the value in scaled form even when it
exceeds the range of the scalar type. Complex arguments are evaluated through
[`faddeeva`](crate::faddeeva).

# Examples

```
// erfi(x) = (2/√π) exp(x²) D(x)
let x = 0.75_f64;
let d = scioxide_special::dawson(x)?;
let expected = core::f64::consts::FRAC_2_SQRT_PI * (x * x).exp() * d;
assert!((scioxide_special::erfi(x)? - expected).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```

[`erfi_scoped`](crate::erfi_scoped) accepts complex arguments only:

```compile_fail
let _ = scioxide_special::erfi_scoped(
    0.75_f64,
    scioxide_special::ErrorFunctionOptions::default(),
    scioxide_core::ExecutionContext::sequential(),
);
```"###, r###"
No uniform ulp target. For real arguments the detailed error bound combines the error of
the Dawson kernel, the final multiplication and the rounding of `x²`. Complex arguments
have no uniform ulp target; the detailed form reports an error estimate. See the [crate
accuracy table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if the value exceeds the range of the scalar type: for real
  `x`, about `|x| > 26.7` in `f64`. [`erfi_with_options`](crate::erfi_with_options) still returns
  the value in scaled form.
- `NumericalInstability` (`LossOfSignificance`) if, for complex `z`, `Re z · Im z` overflows.

# References

- NIST Digital Library of Mathematical Functions, §7.2. <https://dlmf.nist.gov/7.2>"###)]
    #[inline]
    pub erfi [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::ErrorFunctionOptions;
    detailed erfi_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::error_probability::real_api::erfi(x.real_component()).map(T::from_real)
    } complex {
        crate::error_probability::complex_error::erfi_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::error_probability::real_api::erfi_evaluation(x.real_component(), options, context)
    } complex {
        crate::error_probability::complex_error::erfi_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    complex scoped pub erfi_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Dawson's integral D(z).

```text
D(z) = exp(−z²) ∫₀ᶻ exp(t²) dt
```

(DLMF 7.2.5). D is entire and odd, preserves the sign of zero, and behaves like `1/(2x)` as
`x → ∞`. Its derivative is `D′(z) = 1 − 2z D(z)`; the detailed form evaluates it without
cancellation in the asymptotic region.

Real arguments use a series in `x²` for `|x| < 1/2`, piecewise degree-9 Chebyshev interpolants
(sixteen per binade) on `[1/2, 8)`, and an asymptotic expansion in `1/x²` beyond 8. The cost is
fixed. Complex arguments are evaluated through [`faddeeva`](crate::faddeeva).

# Examples

```
// D(2) = 0.30134038892379197, and D is odd
assert!((scioxide_special::dawson(2.0_f64)? - 0.301_340_388_923_791_97).abs() < 1e-14);
let z = scioxide_core::Complex::new(0.25_f64, 0.5);
assert!((scioxide_special::dawson(-z)? + scioxide_special::dawson(z)?).norm() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested
range. The detailed form reports a bound of 2 units for `f64` and 1 unit for `f32`
([error estimates](crate#error-estimates)). Complex arguments have no uniform ulp target;
the detailed form reports an error estimate. See the [crate accuracy
table](crate#accuracy)."###, r###"
Real arguments have no error conditions of their own. For complex `z`:

- `NumericalInstability` (`Overflow`) if a complex value exceeds the range of the scalar type.
- `NumericalInstability` (`LossOfSignificance`) if `Re z · Im z` overflows, so the phase of
  `exp(−z²)` cannot be formed.

# References

- NIST Digital Library of Mathematical Functions, §7.2. <https://dlmf.nist.gov/7.2>"###)]
    #[inline]
    pub dawson [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::ErrorFunctionOptions;
    detailed dawson_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::error_probability::real_api::dawson(x.real_component()).map(T::from_real)
    } complex {
        crate::error_probability::complex_error::dawson_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::error_probability::real_api::dawson_evaluation(x.real_component(), options, context)
    } complex {
        crate::error_probability::complex_error::dawson_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    complex scoped pub dawson_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the natural logarithm of the complementary error function, ln erfc(z).

For real `x`, `erfc(x) > 0` and the result is real. It is computed as `ln(1 − erf(x))` through
`ln1p` for `x < 1/2` and as `ln erfcx(x) − x²` otherwise, so it stays accurate where `erfc(x)`
underflows. For complex `z` the result is the principal logarithm of `erfc(z)`, with its branch
cut where `erfc(z)` lies on the negative real axis.

# Examples

```
// erfc(30) ≈ 2.6e−393 underflows, but ln erfc(x) = ln erfcx(x) − x² does not
let expected = scioxide_special::erfcx(30.0_f64)?.ln() - 900.0;
assert!((scioxide_special::log_erfc(30.0_f64)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested
range. The detailed form reports a bound of 4 units for `f64` and 1 unit for `f32`
([error estimates](crate#error-estimates)). Complex arguments have no uniform ulp target;
the detailed form reports an error estimate. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if, for real `x`, `x²` overflows (`x > 1.3·10¹⁵⁴` in `f64`).
- `NumericalInstability` (`LossOfSignificance`) if, for complex `z`, `erfc(z)` is too close to zero
  to take its logarithm, or the side of the branch cut cannot be resolved.

# References

- NIST Digital Library of Mathematical Functions, §7.2. <https://dlmf.nist.gov/7.2>"###)]
    #[inline]
    pub log_erfc [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::ErrorFunctionOptions;
    detailed log_erfc_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::error_probability::probability::log_erfc(x.real_component()).map(T::from_real)
    } complex {
        crate::error_probability::complex_error::log_erfc_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::error_probability::probability::log_erfc_evaluation(x.real_component(), options, context)
    } complex {
        crate::error_probability::complex_error::log_erfc_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    complex scoped pub log_erfc_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
