//! Family-owned scalar entry points.
use crate::SpecialFunctionResult;
use scioxide_core::{Complex, NumericScalar};

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Fresnel cosine integral C(z).

```text
C(z) = ∫₀ᶻ cos(πt²/2) dt
```

(DLMF 7.2.7). C is entire and odd, and `C(x) → 1/2` as `x → +∞`. It accepts real and complex
arguments in `f32` or `f64`.

Real arguments use Chebyshev expansions on `[0, 8]` and the asymptotic expansions of the
auxiliary functions `f` and `g` beyond. Complex arguments use the power series for `|z| ≤ 1.5`
and otherwise the relation to the error function,
`C(z) + i S(z) = ((1 + i)/2) erf((√π/2)(1 − i) z)` (DLMF §7.5); values that grow like
`exp(π|Im z²|/2)` are kept in scaled form by the detailed form.

# Examples

```
// C(1) = 0.7798934003768228
let c = scioxide_special::fresnel_c(1.0_f64)?;
assert!((c - 0.779_893_400_376_822_8).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`,
for real and complex arguments. See the [crate accuracy table](crate#accuracy)."###, r###"
Real arguments have no error conditions of their own. For complex `z`:

- `NumericalInstability` (`Overflow`) if `πz²/2`, its sine or cosine, or the value exceeds the range
  of the scalar type.

# References

- NIST Digital Library of Mathematical Functions, §7.2 and §7.5. <https://dlmf.nist.gov/7>"###)]
    #[inline]
    pub fresnel_c [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::IntegralOptions;
    detailed fresnel_c_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::integrals::fresnel_c(x.real_component()).map(T::from_real)
    } complex {
        crate::integrals::complex_fresnel::fresnel_c_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::integrals::fresnel_c_evaluation(x.real_component(), options, context)
    } complex {
        crate::integrals::complex_fresnel::fresnel_c_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub fresnel_c_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Fresnel sine integral S(z).

```text
S(z) = ∫₀ᶻ sin(πt²/2) dt
```

(DLMF 7.2.8). S is entire and odd, and `S(x) → 1/2` as `x → +∞`. It accepts real and complex
arguments in `f32` or `f64`.

Real arguments use Chebyshev expansions on `[0, 8]` and the asymptotic expansions of the
auxiliary functions `f` and `g` beyond. Complex arguments use the power series for `|z| ≤ 1.5`
and otherwise the relation to the error function,
`C(z) + i S(z) = ((1 + i)/2) erf((√π/2)(1 − i) z)` (DLMF §7.5); values that grow like
`exp(π|Im z²|/2)` are kept in scaled form by the detailed form.

# Examples

```
// S(1) = 0.4382591473903548
let s = scioxide_special::fresnel_s(1.0_f64)?;
assert!((s - 0.438_259_147_390_354_8).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`,
for real and complex arguments. See the [crate accuracy table](crate#accuracy)."###, r###"
Real arguments have no error conditions of their own. For complex `z`:

- `NumericalInstability` (`Overflow`) if `πz²/2`, its sine or cosine, or the value exceeds the range
  of the scalar type.

# References

- NIST Digital Library of Mathematical Functions, §7.2 and §7.5. <https://dlmf.nist.gov/7>"###)]
    #[inline]
    pub fresnel_s [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::IntegralOptions;
    detailed fresnel_s_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::integrals::fresnel_s(x.real_component()).map(T::from_real)
    } complex {
        crate::integrals::complex_fresnel::fresnel_s_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::integrals::fresnel_s_evaluation(x.real_component(), options, context)
    } complex {
        crate::integrals::complex_fresnel::fresnel_s_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub fresnel_s_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
