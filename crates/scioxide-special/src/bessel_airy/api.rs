//! Plain, detailed and scoped entry points for Bessel, Hankel and Airy functions.
use crate::SpecialFunctionResult;
use scioxide_core::{Complex, NumericScalar};
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Bessel function of the first kind `J_ν(z)`.

```text
J_ν(z) = (z/2)^ν Σₖ₌₀^∞ (−z²/4)^k / (k! Γ(ν + k + 1))      (DLMF 10.2.2)
```

`ν` is the order and `z` the argument. `ν` may be real while `z` is complex. Complex
values use the principal branch, with a cut along the negative real axis; on the cut, the sign
of the zero imaginary part of `z` selects the upper or lower side. For real `x < 0`, integer
orders use `J_n(−x) = (−1)^n J_n(x)`. At the origin `J_0(0) = 1`, and `J_ν(0) = 0` for integer
`ν ≠ 0` and for `ν > 0` (`Re ν > 0`).

Real arguments use the power series for `x ≤ 2`, Steed's continued-fraction method for larger
`x`, and Hankel's asymptotic expansion (DLMF §10.17) where its terms reach `f64` accuracy.
Orders of at least 1000 with `ν > 2√x` use the uniform asymptotic expansions of DLMF
§10.19–10.20. Orders 0 and 1 use rational approximations, with Taylor polynomials about each
zero for `8 < x < 256`. Orders below `−1/2` use the reflection formulas of DLMF §10.4. Complex
arguments and orders use the power series and Temme's series near the origin, Miller's backward
recurrence normalized by the Wronskian at moderate `|z|`, Hankel's expansion for large `|z|`,
and Debye's expansion for `|ν| ≥ 64`. The work is bounded by the iteration budget of
[`BesselOptions`](crate::BesselOptions). For real `x > 0` and `ν ≥ −1/2`,
[`bessel_j_with_options`](crate::bessel_j_with_options) also returns `J_ν′(x)` when it is
representable.

# Examples

```
// J₀(1), from the series DLMF 10.2.2
let j0 = scioxide_special::bessel_j(0.0_f64, 1.0)?;
assert!((j0 - 0.7651976865579666).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `ν` and `x`, the error is within 64 ulps for `f64` and `f32` on the tested
range, which includes `x` up to `10⁸` with orders below 1000 and orders near `√x`. Near a zero
of the function the tests bound the absolute error instead. Complex arguments and complex orders
have no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0` when `ν` is not an integer.
- `DomainError` (`Pole`) at `z = 0` when `ν` is a negative non-integer, or complex with
  `Re ν ≤ 0`.
- `NumericalInstability` (`Overflow`) if `|J_ν(z)|` exceeds the range of the scalar type, as for
  large `|Im z|`; [`bessel_j_scaled`](crate::bessel_j_scaled) avoids this.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>
- I. J. Thompson and A. R. Barnett, "Coulomb and Bessel functions of complex arguments and
  order", *J. Comput. Phys.* 64 (1986), 490–509."###)]
    #[inline]
    pub bessel_j [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_j_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::bessel_j(order.real_component(), z.real_component())
            .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_j_complex(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::bessel_j_evaluate(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_j_complex_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub bessel_j_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Bessel function of the second kind `Y_ν(z)`.

```text
Y_ν(z) = (J_ν(z) cos(νπ) − J_(−ν)(z)) / sin(νπ)      (DLMF 10.2.3)
```

with the limit `ν → n` at integer orders. `ν` is the order and `z` the argument. `ν` may be real while
`z` is complex. Complex values use the principal branch, with a cut along the
negative real axis; on the cut, the sign of the zero imaginary part of `z` selects the upper or
lower side. Real arguments must be positive.

Real arguments use Temme's series for `0 < x ≤ 2`, Steed's continued-fraction method for larger
`x`, and Hankel's asymptotic expansion (DLMF §10.17) where its terms reach `f64` accuracy.
Orders of at least 1000 with `ν > 2√x` use the uniform asymptotic expansions of DLMF
§10.19–10.20. Orders 0 and 1 use rational approximations, with Taylor polynomials about each
zero for `8 < x < 256`. Orders below `−1/2` use the reflection formulas of DLMF §10.4. Complex
arguments and orders use Temme's series near the origin, which removes the apparent singularity
at integer orders, and scaled recurrences and connection formulas for larger `|z|`. The work is
bounded by the iteration budget of [`BesselOptions`](crate::BesselOptions). For real `ν ≥ −1/2`,
[`bessel_y_with_options`](crate::bessel_y_with_options) also returns `Y_ν′(x)` when it is
representable.

# Examples

```
// Y₀(1), from the series DLMF 10.8.2
let y0 = scioxide_special::bessel_y(0.0_f64, 1.0)?;
assert!((y0 - 0.08825696421567696).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `ν` and `x`, the error is within 64 ulps for `f64` and `f32` on the tested
range, which includes `x` up to `10⁸` with orders below 1000 and orders near `√x`. Near a zero
of the function the tests bound the absolute error instead. Complex arguments and complex orders
have no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `NumericalInstability` (`Overflow`) if `|Y_ν(z)|` exceeds the range of the scalar type, as
  near the origin for large `|ν|`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>
- N. M. Temme, "On the numerical evaluation of the ordinary Bessel function of the second kind",
  *J. Comput. Phys.* 21 (1976), 343–350.
- I. J. Thompson and A. R. Barnett, "Coulomb and Bessel functions of complex arguments and
  order", *J. Comput. Phys.* 64 (1986), 490–509."###)]
    #[inline]
    pub bessel_y [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_y_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::bessel_y(order.real_component(), z.real_component())
            .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_y_complex(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::bessel_y_evaluate(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_y_complex_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub bessel_y_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the modified Bessel function of the first kind `I_ν(z)`.

```text
I_ν(z) = (z/2)^ν Σₖ₌₀^∞ (z²/4)^k / (k! Γ(ν + k + 1))      (DLMF 10.25.2)
```

`ν` is the order and `z` the argument. `ν` may be real while `z` is complex. Complex
values use the principal branch, with a cut along the negative real axis; on the cut, the sign
of the zero imaginary part of `z` selects the upper or lower side. For real `x < 0`, integer
orders use `I_n(−x) = (−1)^n I_n(x)`. At the origin `I_0(0) = 1`, and `I_ν(0) = 0` for integer
`ν ≠ 0` and for `ν > 0` (`Re ν > 0`).

Real arguments use the power series, or Hankel's asymptotic expansion (DLMF §10.40) for large
`x`; orders with `|ν| ≥ 32` use Debye's uniform expansion (DLMF §10.41). Negative non-integer
orders use `I_(−ν) = I_ν + (2/π) sin(νπ) K_ν` (DLMF §10.27). Complex arguments and orders use
the power series and Temme's series near the origin, Miller's backward recurrence normalized by
the Wronskian at moderate `|z|`, Hankel's expansion for large `|z|`, and Debye's expansion for
`|ν| ≥ 64`. The work is bounded by the iteration budget of
[`BesselOptions`](crate::BesselOptions).

# Examples

```
// I₀(1), from the series DLMF 10.25.2
let i0 = scioxide_special::bessel_i(0.0_f64, 1.0)?;
assert!((i0 - 1.2660658777520082).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `ν` and `x`, the error is within 64 ulps for `f64` and `f32` on the tested
range. Complex arguments and complex orders have no uniform ulp bound; the detailed form reports
an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0` when `ν` is not an integer.
- `DomainError` (`Pole`) at `z = 0` when `ν` is a negative non-integer, or complex with
  `Re ν ≤ 0`.
- `NumericalInstability` (`Overflow`) if `|I_ν(z)|` exceeds the range of the scalar type, as for
  large `|Re z|`; [`bessel_i_scaled`](crate::bessel_i_scaled) avoids this.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>
- I. J. Thompson and A. R. Barnett, "Coulomb and Bessel functions of complex arguments and
  order", *J. Comput. Phys.* 64 (1986), 490–509."###)]
    #[inline]
    pub bessel_i [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_i_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::bessel_i(order.real_component(), z.real_component())
            .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_i_complex(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::bessel_i_evaluate(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_i_complex_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub bessel_i_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the modified Bessel function of the second kind `K_ν(z)`.

```text
K_ν(z) = π (I_(−ν)(z) − I_ν(z)) / (2 sin(νπ))      (DLMF 10.27.4)
```

with the limit `ν → n` at integer orders; `K_(−ν) = K_ν`. `ν` is the order and `z` the argument.
`ν` may be real while `z` is complex. Complex values use the principal branch, with a cut
along the negative real axis; on the cut, the sign of the zero imaginary part of `z` selects the
upper or lower side. Real arguments must be nonnegative. At real `x = 0` the function returns
`+∞`, the limit of `K_ν(x)` as `x → 0⁺`, for every finite order; the detailed form reports it as
an infinite limit (see [detailed results](crate#detailed-results)).

Real arguments use Temme's series for `x ≤ 2`, Steed's continued fraction for `2 < x < 40` and
Hankel's asymptotic expansion (DLMF §10.40) beyond, each followed by forward recurrence in the
order; orders with `|ν| ≥ 32` use Debye's uniform expansion (DLMF §10.41). Complex arguments and
orders use the power series and Temme's series near the origin, Miller's backward recurrence
normalized by the Wronskian at moderate `|z|`, Hankel's expansion for large `|z|`, and Debye's
expansion for `|ν| ≥ 64`. The work is bounded by the iteration budget of
[`BesselOptions`](crate::BesselOptions).

# Examples

```
// K₀(1), from the series DLMF 10.31.2
let k0 = scioxide_special::bessel_k(0.0_f64, 1.0)?;
assert!((k0 - 0.42102443824070834).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `ν` and `x`, the error is within 64 ulps for `f64` and `f32` on the tested
range. Complex arguments and complex orders have no uniform ulp bound; the detailed form reports
an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `DomainError` (`Pole`) at complex `z = 0`; real `x = 0` returns `+∞` instead.
- `NumericalInstability` (`Overflow`) if `|K_ν(z)|` exceeds the range of the scalar type, as
  near the origin for large `|ν|`; [`bessel_k_log`](crate::bessel_k_log) returns `ln K_ν(x)` for
  real arguments.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>
- N. M. Temme, "On the numerical evaluation of the modified Bessel function of the third kind",
  *J. Comput. Phys.* 19 (1975), 324–337.
- I. J. Thompson and A. R. Barnett, "Coulomb and Bessel functions of complex arguments and
  order", *J. Comput. Phys.* 64 (1986), 490–509."###)]
    #[inline]
    pub bessel_k [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_k_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::bessel_k(order.real_component(), z.real_component())
            .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_k_complex(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::bessel_k_evaluate(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_k_complex_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub bessel_k_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled modified Bessel function `e^(−|Re z|) I_ν(z)`.

For real arguments the factor is `e^(−|x|)`. The scaled value stays in range for large `|Re z|`,
where `I_ν(z)` grows like `e^|Re z|`. Orders, domain, branch cut and origin values are those of
[`bessel_i`](crate::bessel_i). Real order 0 uses Chebyshev expansions on `[0, 8]` and `[8, ∞)`;
other orders use the methods of [`bessel_i`](crate::bessel_i).

`ν` may be real while `z` is complex.

# Examples

```
// e^(−1) I₀(1)
let scaled = scioxide_special::bessel_i_scaled(0.0_f64, 1.0)?;
let expected = (-1.0_f64).exp() * scioxide_special::bessel_i(0.0_f64, 1.0)?;
assert!((scaled - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `ν` and `x`, the error is within 64 ulps for `f64` and `f32` on the tested
range. Complex arguments and complex orders have no uniform ulp bound; the detailed form reports
an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0` when `ν` is not an integer.
- `DomainError` (`Pole`) at `z = 0` when `ν` is a negative non-integer, or complex with
  `Re ν ≤ 0`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub bessel_i_scaled [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_i_scaled_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::bessel_i_scaled(order.real_component(), z.real_component())
            .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_i_complex_scaled(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::bessel_i_scaled_evaluate(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_i_complex_scaled_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub bessel_i_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled modified Bessel function `e^z K_ν(z)`.

For real arguments the factor is `e^x`; for complex `z` it includes the phase `e^(i Im z)`. The
scaled value stays in range for large `Re z`, where `K_ν(z)` decays like `e^(−z)`. Orders,
domain, branch cut and the `+∞` limit at real `x = 0` are those of
[`bessel_k`](crate::bessel_k).

`ν` may be real while `z` is complex.

# Examples

```
// e¹ K₀(1)
let scaled = scioxide_special::bessel_k_scaled(0.0_f64, 1.0)?;
let expected = 1.0_f64.exp() * scioxide_special::bessel_k(0.0_f64, 1.0)?;
assert!((scaled - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `ν` and `x`, the error is within 64 ulps for `f64` and `f32` on the tested
range. Complex arguments and complex orders have no uniform ulp bound; the detailed form reports
an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `DomainError` (`Pole`) at complex `z = 0`; real `x = 0` returns `+∞` instead.
- `NumericalInstability` (`Overflow`) if the scaled value exceeds the range of the scalar type,
  as near the origin for large `|ν|`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub bessel_k_scaled [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_k_scaled_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::bessel_k_scaled(order.real_component(), z.real_component())
            .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_k_complex_scaled(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::bessel_k_scaled_evaluate(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_k_complex_scaled_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub bessel_k_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the spherical Bessel function of the first kind `j_n(z)`.

```text
j_n(z) = √(π/(2z)) J_(n+1/2)(z)      (DLMF 10.47.3)
```

The argument `z` is real or complex and the order `n` is a nonnegative integer. Complex values
are formed from the principal branches of `√z` and of the cylindrical function. `j_n` is entire,
with `j_n(−z) = (−1)^n j_n(z)`, `j_0(0) = 1` and `j_n(0) = 0` for `n ≥ 1`.

Real arguments use the power series for `|x| ≤ 1` and otherwise a recurrence in the order:
Miller's backward recurrence where forward recurrence would be unstable, forward recurrence
otherwise. Complex arguments use the power series for `|z| < 1/4` and otherwise the cylindrical
function of order `n + 1/2`. The work is bounded by the iteration budget of
[`SphericalBesselOptions`](crate::SphericalBesselOptions).

# Examples

```
// j₀(x) = sin(x)/x (DLMF 10.49.3)
let j0 = scioxide_special::spherical_jn(0, 1.0_f64)?;
assert!((j0 - 1.0_f64.sin()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|j_n(z)|` exceeds the range of the scalar
  type, as for complex `z` with large `|Im z|`.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]
    #[inline]
    pub spherical_jn [T: NumericScalar] (order: u32, z: T) -> T
    where [];
    options options: crate::SphericalBesselOptions;
    detailed spherical_jn_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::spherical::spherical_jn(order, z.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::spherical_jn_complex(order, crate::dispatch::complex(z))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::spherical::spherical_jn_evaluation(
            order,
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::spherical_jn_complex_evaluation(
            order,
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub spherical_jn_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the spherical Bessel function of the second kind `y_n(z)`.

```text
y_n(z) = √(π/(2z)) Y_(n+1/2)(z)      (DLMF 10.47.4)
```

The argument `z` is real or complex and the order `n` is a nonnegative integer. Complex values
are formed from the principal branches of `√z` and of the cylindrical function. `y_n` has a pole
at `z = 0` and satisfies `y_n(−z) = (−1)^(n+1) y_n(z)`, so negative real arguments are accepted.

Real arguments use forward recurrence in the order. Complex arguments use the power series for
`|z| < 1/4` and otherwise the cylindrical function of order `n + 1/2`. The work is bounded by
the iteration budget of [`SphericalBesselOptions`](crate::SphericalBesselOptions).

# Examples

```
// y₀(x) = −cos(x)/x (DLMF 10.49.4)
let y0 = scioxide_special::spherical_yn(0, 1.0_f64)?;
assert!((y0 + 1.0_f64.cos()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `NumericalInstability` (`Overflow`) if `|y_n(z)|` exceeds the range of the scalar type, as
  near the origin for large `n`.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]
    #[inline]
    pub spherical_yn [T: NumericScalar] (order: u32, z: T) -> T
    where [];
    options options: crate::SphericalBesselOptions;
    detailed spherical_yn_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::spherical::spherical_yn(order, z.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::spherical_yn_complex(order, crate::dispatch::complex(z))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::spherical::spherical_yn_evaluation(
            order,
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::spherical_yn_complex_evaluation(
            order,
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub spherical_yn_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the modified spherical Bessel function of the first kind `i_n(z)`.

```text
i_n(z) = √(π/(2z)) I_(n+1/2)(z)
```

This is `i⁽¹⁾_n` of DLMF §10.47. The argument `z` is real or complex and the order `n` is a
nonnegative integer. Complex values are formed from the principal branches of `√z` and of the
cylindrical function. `i_n` is entire, with `i_n(−z) = (−1)^n i_n(z)`, `i_0(0) = 1` and
`i_n(0) = 0` for `n ≥ 1`.

Real arguments use the power series for `|x| ≤ 1` and otherwise a recurrence in the order:
Miller's backward recurrence where forward recurrence would be unstable, forward recurrence
otherwise. Complex arguments use the power series for `|z| < 1/4` and otherwise the cylindrical
function of order `n + 1/2`. The work is bounded by the iteration budget of
[`SphericalBesselOptions`](crate::SphericalBesselOptions).

# Examples

```
// i₀(x) = sinh(x)/x (DLMF §10.49)
let i0 = scioxide_special::spherical_in(0, 1.0_f64)?;
assert!((i0 - 1.0_f64.sinh()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|i_n(z)|` exceeds the range of the scalar
  type, as for large `|Re z|`; [`spherical_in_scaled`](crate::spherical_in_scaled) avoids this.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]
    #[inline]
    pub spherical_in [T: NumericScalar] (order: u32, z: T) -> T
    where [];
    options options: crate::SphericalBesselOptions;
    detailed spherical_in_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::spherical::spherical_in(order, z.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::spherical_in_complex(order, crate::dispatch::complex(z))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::spherical::spherical_in_evaluation(
            order,
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::spherical_in_complex_evaluation(
            order,
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub spherical_in_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the modified spherical Bessel function of the second kind `k_n(z)`.

```text
k_n(z) = √(π/(2z)) K_(n+1/2)(z)
```

as in DLMF §10.47, so `k_0(z) = π e^(−z)/(2z)`. The argument `z` is real or complex and the
order `n` is a nonnegative integer. Real arguments must be positive; `k_n` has a pole at
`z = 0`.

Real arguments use forward recurrence in the order. Complex arguments use the power series for
`|z| < 1/4` and otherwise the cylindrical function of order `n + 1/2`. The work is bounded by
the iteration budget of [`SphericalBesselOptions`](crate::SphericalBesselOptions).

# Examples

```
// k₀(x) = π e^(−x)/(2x) (DLMF §10.49)
let k0 = scioxide_special::spherical_kn(0, 1.0_f64)?;
let expected = core::f64::consts::FRAC_PI_2 * (-1.0_f64).exp();
assert!((k0 - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `NumericalInstability` (`Overflow`) if `|k_n(z)|` exceeds the range of the scalar type, as
  near the origin for large `n`.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]
    #[inline]
    pub spherical_kn [T: NumericScalar] (order: u32, z: T) -> T
    where [];
    options options: crate::SphericalBesselOptions;
    detailed spherical_kn_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::spherical::spherical_kn(order, z.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::spherical_kn_complex(order, crate::dispatch::complex(z))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::spherical::spherical_kn_evaluation(
            order,
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::spherical_kn_complex_evaluation(
            order,
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub spherical_kn_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled modified spherical Bessel function `e^(−|Re z|) i_n(z)`.

For real arguments the factor is `e^(−|x|)`. The scaled value stays in range for large `|Re z|`.
Orders, domain and origin values are those of [`spherical_in`](crate::spherical_in).

# Examples

```
// e^(−1) i₀(1) = e^(−1) sinh(1)
let scaled = scioxide_special::spherical_in_scaled(0, 1.0_f64)?;
assert!((scaled - (-1.0_f64).exp() * 1.0_f64.sinh()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"No conditions beyond those above.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]
    #[inline]
    pub spherical_in_scaled [T: NumericScalar] (order: u32, z: T) -> T
    where [];
    options options: crate::SphericalBesselOptions;
    detailed spherical_in_scaled_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::spherical::spherical_in_scaled(order, z.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::spherical_in_complex_scaled(
            order,
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::spherical::spherical_in_scaled_evaluation(
            order,
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::spherical_in_complex_scaled_evaluation(
            order,
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub spherical_in_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled modified spherical Bessel function `e^z k_n(z)`.

For real arguments the factor is `e^x`; for complex `z` it includes the phase `e^(i Im z)`. The
scaled value stays in range for large `Re z`. Orders, domain and normalization are those of
[`spherical_kn`](crate::spherical_kn).

# Examples

```
// e¹ k₀(1) = π/2 (DLMF §10.49)
let scaled = scioxide_special::spherical_kn_scaled(0, 1.0_f64)?;
assert!((scaled - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `NumericalInstability` (`Overflow`) if the scaled value exceeds the range of the scalar type,
  as near the origin for large `n`.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]
    #[inline]
    pub spherical_kn_scaled [T: NumericScalar] (order: u32, z: T) -> T
    where [];
    options options: crate::SphericalBesselOptions;
    detailed spherical_kn_scaled_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::spherical::spherical_kn_scaled(order, z.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::spherical_kn_complex_scaled(
            order,
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::spherical::spherical_kn_scaled_evaluation(
            order,
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::spherical_kn_complex_scaled_evaluation(
            order,
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub spherical_kn_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Hankel function of the first kind `H⁽¹⁾_ν(z) = J_ν(z) + i Y_ν(z)`.

The definition is DLMF 10.4.3. The result is complex for real and complex inputs. `ν` is the
order and `z` the argument. `ν` may be real while `z` is complex. Complex values use the
principal branch, with a cut along the negative real axis; on the cut, the sign of the zero
imaginary part of `z` selects the upper or lower side. On the imaginary axis (`Re z = ±0`,
`Im z ≠ 0`) both signs of the zero real part give the same value. Real arguments must be
positive.

For real `x > 0` the real part is [`bessel_j`](crate::bessel_j) and the imaginary part is
[`bessel_y`](crate::bessel_y). Complex arguments and orders use the complex kernels described at
[`bessel_j`](crate::bessel_j).

# Examples

```
use scioxide_core::Complex;
use scioxide_special::{bessel_j, bessel_y, hankel_1};

// H⁽¹⁾₀(1) = J₀(1) + i Y₀(1)
let expected = Complex::new(bessel_j(0.0_f64, 1.0)?, bessel_y(0.0_f64, 1.0)?);
assert!((hankel_1(0.0_f64, 1.0)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `ν` and `x > 0`, the error is within 64 ulps for `f64` and `f32` on the
tested range. Complex arguments and complex orders have no uniform ulp bound; the detailed form
reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `NumericalInstability` (`Overflow`) if `|H⁽¹⁾_ν(z)|` exceeds the range of the scalar type, as
  near the origin for large `|ν|` or for large negative `Im z`;
  [`hankel_1_scaled`](crate::hankel_1_scaled) avoids the latter.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>
- I. J. Thompson and A. R. Barnett, "Coulomb and Bessel functions of complex arguments and
  order", *J. Comput. Phys.* 64 (1986), 490–509."###)]
    #[inline]
    pub hankel_1 [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> Complex<T::Real>
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed hankel_1_with_options -> SpecialFunctionResult<Complex<T::Real>>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::hankel_1(order.real_component(), z.real_component())
    } complex {
        crate::bessel_airy::complex_bessel::hankel_1_complex(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
        )
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::hankel_1_evaluation(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::hankel_1_complex_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub hankel_1_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Hankel function of the second kind `H⁽²⁾_ν(z) = J_ν(z) − i Y_ν(z)`.

The definition is DLMF 10.4.3. The result is complex for real and complex inputs. `ν` is the
order and `z` the argument. `ν` may be real while `z` is complex. Complex values use the
principal branch, with a cut along the negative real axis; on the cut, the sign of the zero
imaginary part of `z` selects the upper or lower side. On the imaginary axis (`Re z = ±0`,
`Im z ≠ 0`) both signs of the zero real part give the same value. Real arguments must be
positive.

For real `x > 0` the real part is [`bessel_j`](crate::bessel_j) and the imaginary part is the
negative of [`bessel_y`](crate::bessel_y). Complex arguments and orders use the complex kernels
described at [`bessel_j`](crate::bessel_j).

# Examples

```
use scioxide_core::Complex;
use scioxide_special::{bessel_j, bessel_y, hankel_2};

// H⁽²⁾₀(1) = J₀(1) − i Y₀(1)
let expected = Complex::new(bessel_j(0.0_f64, 1.0)?, -bessel_y(0.0_f64, 1.0)?);
assert!((hankel_2(0.0_f64, 1.0)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `ν` and `x > 0`, the error is within 64 ulps for `f64` and `f32` on the
tested range. Complex arguments and complex orders have no uniform ulp bound; the detailed form
reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `NumericalInstability` (`Overflow`) if `|H⁽²⁾_ν(z)|` exceeds the range of the scalar type, as
  near the origin for large `|ν|` or for large positive `Im z`;
  [`hankel_2_scaled`](crate::hankel_2_scaled) avoids the latter.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>
- I. J. Thompson and A. R. Barnett, "Coulomb and Bessel functions of complex arguments and
  order", *J. Comput. Phys.* 64 (1986), 490–509."###)]
    #[inline]
    pub hankel_2 [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> Complex<T::Real>
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed hankel_2_with_options -> SpecialFunctionResult<Complex<T::Real>>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::hankel_2(order.real_component(), z.real_component())
    } complex {
        crate::bessel_airy::complex_bessel::hankel_2_complex(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
        )
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::hankel_2_evaluation(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::hankel_2_complex_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub hankel_2_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the spherical Hankel function of the first kind `h⁽¹⁾_n(z) = j_n(z) + i y_n(z)`.

The result is complex for real and complex inputs. The argument `z` is real or complex and the
order `n` is a nonnegative integer. Complex values are formed from the principal branches of
`√z` and of the cylindrical function. `h⁽¹⁾_n` has a pole at `z = 0`; negative real arguments
are accepted. Off the real axis the function is evaluated through the modified function of order
`n + 1/2`.

# Examples

```
use scioxide_core::Complex;

// h⁽¹⁾₀(x) = (sin(x) − i cos(x))/x (DLMF §10.49)
let h = scioxide_special::spherical_hankel_1(0, 1.0_f64)?;
assert!((h - Complex::new(1.0_f64.sin(), -1.0_f64.cos())).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `NumericalInstability` (`Overflow`) if `|h⁽¹⁾_n(z)|` exceeds the range of the scalar type, as
  near the origin for large `n`.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]
    #[inline]
    pub spherical_hankel_1 [T: NumericScalar] (order: u32, z: T) -> Complex<T::Real>
    where [];
    options options: crate::SphericalBesselOptions;
    detailed spherical_hankel_1_with_options -> SpecialFunctionResult<Complex<T::Real>>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::spherical_hankel_1(order, z.real_component())
    } complex {
        crate::bessel_airy::complex_bessel::spherical_hankel_1_complex(
            order,
            crate::dispatch::complex(z),
        )
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::spherical_hankel_1_evaluation(
            order,
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::spherical_hankel_1_complex_evaluation(
            order,
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub spherical_hankel_1_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the spherical Hankel function of the second kind `h⁽²⁾_n(z) = j_n(z) − i y_n(z)`.

The result is complex for real and complex inputs. The argument `z` is real or complex and the
order `n` is a nonnegative integer. Complex values are formed from the principal branches of
`√z` and of the cylindrical function. `h⁽²⁾_n` has a pole at `z = 0`; negative real arguments
are accepted. Off the real axis the function is evaluated through the modified function of order
`n + 1/2`.

# Examples

```
use scioxide_core::Complex;

// h⁽²⁾₀(x) = (sin(x) + i cos(x))/x (DLMF §10.49)
let h = scioxide_special::spherical_hankel_2(0, 1.0_f64)?;
assert!((h - Complex::new(1.0_f64.sin(), 1.0_f64.cos())).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `NumericalInstability` (`Overflow`) if `|h⁽²⁾_n(z)|` exceeds the range of the scalar type, as
  near the origin for large `n`.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]
    #[inline]
    pub spherical_hankel_2 [T: NumericScalar] (order: u32, z: T) -> Complex<T::Real>
    where [];
    options options: crate::SphericalBesselOptions;
    detailed spherical_hankel_2_with_options -> SpecialFunctionResult<Complex<T::Real>>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::spherical_hankel_2(order, z.real_component())
    } complex {
        crate::bessel_airy::complex_bessel::spherical_hankel_2_complex(
            order,
            crate::dispatch::complex(z),
        )
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::spherical_hankel_2_evaluation(
            order,
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::spherical_hankel_2_complex_evaluation(
            order,
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub spherical_hankel_2_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `k`-th argument derivative of [`bessel_j`](crate::bessel_j), where `k` is `degree`.

```text
J_ν⁽ᵏ⁾(z) = 2^(−k) Σₘ₌₀^k (−1)^m C(k, m) J_(ν−k+2m)(z)      (DLMF §10.6)
```

`C(k, m)` is the binomial coefficient and the order `ν` is held fixed; `k = 0` returns the
function itself. Orders, domain and branch cut are those of [`bessel_j`](crate::bessel_j). Each
term is computed with the same kernels, so the cost grows linearly with `k`.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_special::{bessel_j, bessel_j_derivative};

// J₀′(x) = −J₁(x) (DLMF §10.6)
let expected = -bessel_j(1.0_f64, 1.0)?;
assert!((bessel_j_derivative(0.0_f64, 1.0, 1)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Derivatives have no uniform ulp bound. The detailed form reports an error estimate
that includes the rounding of the binomial sum. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0` when `ν` is not an integer.
- `DomainError` (`Pole`) at `z = 0` when `ν` is not an integer and `ν < k`, or when `ν` is
  complex with `Re ν ≤ k`; there the derivative is unbounded.
- `ConvergenceFailure` (`IterationLimit`) if `k` exceeds the remaining iteration budget.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub bessel_j_derivative [T: NumericScalar] (order: impl crate::Parameter<T>, z: T, degree: u32) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_j_derivative_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::bessel_j_derivative(
            order.real_component(),
            z.real_component(),
            degree,
        )
        .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_j_complex_derivative(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::bessel_j_derivative_evaluate(
            order.real_component(),
            z.real_component(),
            degree,
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_j_complex_derivative_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
            options,
            context,
        )
    }
    scoped pub bessel_j_derivative_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `k`-th argument derivative of [`bessel_y`](crate::bessel_y), where `k` is `degree`.

```text
Y_ν⁽ᵏ⁾(z) = 2^(−k) Σₘ₌₀^k (−1)^m C(k, m) Y_(ν−k+2m)(z)      (DLMF §10.6)
```

`C(k, m)` is the binomial coefficient and the order `ν` is held fixed; `k = 0` returns the
function itself. Orders, domain and branch cut are those of [`bessel_y`](crate::bessel_y). Each
term is computed with the same kernels, so the cost grows linearly with `k`.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_special::{bessel_y, bessel_y_derivative};

// Y₀′(x) = −Y₁(x) (DLMF §10.6)
let expected = -bessel_y(1.0_f64, 1.0)?;
assert!((bessel_y_derivative(0.0_f64, 1.0, 1)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Derivatives have no uniform ulp bound. The detailed form reports an error estimate
that includes the rounding of the binomial sum. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`, also for `k = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `ConvergenceFailure` (`IterationLimit`) if `k` exceeds the remaining iteration budget.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub bessel_y_derivative [T: NumericScalar] (order: impl crate::Parameter<T>, z: T, degree: u32) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_y_derivative_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::bessel_y_derivative(
            order.real_component(),
            z.real_component(),
            degree,
        )
        .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_y_complex_derivative(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::bessel_y_derivative_evaluate(
            order.real_component(),
            z.real_component(),
            degree,
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_y_complex_derivative_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
            options,
            context,
        )
    }
    scoped pub bessel_y_derivative_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `k`-th argument derivative of [`bessel_i`](crate::bessel_i), where `k` is `degree`.

```text
I_ν⁽ᵏ⁾(z) = 2^(−k) Σₘ₌₀^k C(k, m) I_(ν−k+2m)(z)      (DLMF §10.29)
```

`C(k, m)` is the binomial coefficient and the order `ν` is held fixed; `k = 0` returns the
function itself. Orders, domain and branch cut are those of [`bessel_i`](crate::bessel_i). Each
term is computed with the same kernels, so the cost grows linearly with `k`.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_special::{bessel_i, bessel_i_derivative};

// I₀′(x) = I₁(x) (DLMF §10.29)
let expected = bessel_i(1.0_f64, 1.0)?;
assert!((bessel_i_derivative(0.0_f64, 1.0, 1)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Derivatives have no uniform ulp bound. The detailed form reports an error estimate
that includes the rounding of the binomial sum. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0` when `ν` is not an integer.
- `DomainError` (`Pole`) at `z = 0` when `ν` is not an integer and `ν < k`, or when `ν` is
  complex with `Re ν ≤ k`; there the derivative is unbounded.
- `ConvergenceFailure` (`IterationLimit`) if `k` exceeds the remaining iteration budget.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub bessel_i_derivative [T: NumericScalar] (order: impl crate::Parameter<T>, z: T, degree: u32) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_i_derivative_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::bessel_i_derivative(
            order.real_component(),
            z.real_component(),
            degree,
        )
        .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_i_complex_derivative(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::bessel_i_derivative_evaluate(
            order.real_component(),
            z.real_component(),
            degree,
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_i_complex_derivative_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
            options,
            context,
        )
    }
    scoped pub bessel_i_derivative_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `k`-th argument derivative of [`bessel_k`](crate::bessel_k), where `k` is `degree`.

```text
K_ν⁽ᵏ⁾(z) = (−1)^k 2^(−k) Σₘ₌₀^k C(k, m) K_(ν−k+2m)(z)      (DLMF §10.29)
```

`C(k, m)` is the binomial coefficient and the order `ν` is held fixed; `k = 0` returns the
function itself. Orders, domain and branch cut are those of [`bessel_k`](crate::bessel_k). Each
term is computed with the same kernels, so the cost grows linearly with `k`.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_special::{bessel_k, bessel_k_derivative};

// K₀′(x) = −K₁(x) (DLMF §10.29)
let expected = -bessel_k(1.0_f64, 1.0)?;
assert!((bessel_k_derivative(0.0_f64, 1.0, 1)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Derivatives have no uniform ulp bound. The detailed form reports an error estimate
that includes the rounding of the binomial sum. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`, also for `k = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `ConvergenceFailure` (`IterationLimit`) if `k` exceeds the remaining iteration budget.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub bessel_k_derivative [T: NumericScalar] (order: impl crate::Parameter<T>, z: T, degree: u32) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_k_derivative_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::bessel_k_derivative(
            order.real_component(),
            z.real_component(),
            degree,
        )
        .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_k_complex_derivative(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::bessel_k_derivative_evaluate(
            order.real_component(),
            z.real_component(),
            degree,
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_k_complex_derivative_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
            options,
            context,
        )
    }
    scoped pub bessel_k_derivative_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `e^(−|Re z|) I_ν⁽ᵏ⁾(z)`, the scaled `k`-th argument derivative of [`bessel_i`](crate::bessel_i), where `k` is `degree`.

```text
I_ν⁽ᵏ⁾(z) = 2^(−k) Σₘ₌₀^k C(k, m) I_(ν−k+2m)(z)      (DLMF §10.29)
```

`C(k, m)` is the binomial coefficient and the order `ν` is held fixed; `k = 0` returns the
function itself. Orders, domain and branch cut are those of
[`bessel_i_scaled`](crate::bessel_i_scaled). Each term is computed with the same kernels, so the
cost grows linearly with `k`.

The result is `e^(−|Re z|)` times the `k`-th derivative of the unscaled function; the factor
itself is not differentiated.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_special::{bessel_i, bessel_i_scaled_derivative};

// e^(−x) I₀′(x) = e^(−x) I₁(x)
let expected = (-1.0_f64).exp() * bessel_i(1.0_f64, 1.0)?;
assert!((bessel_i_scaled_derivative(0.0_f64, 1.0, 1)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Derivatives have no uniform ulp bound. The detailed form reports an error estimate
that includes the rounding of the binomial sum. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0` when `ν` is not an integer.
- `DomainError` (`Pole`) at `z = 0` when `ν` is not an integer and `ν < k`, or when `ν` is
  complex with `Re ν ≤ k`; there the derivative is unbounded.
- `ConvergenceFailure` (`IterationLimit`) if `k` exceeds the remaining iteration budget.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub bessel_i_scaled_derivative [T: NumericScalar] (order: impl crate::Parameter<T>, z: T, degree: u32) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_i_scaled_derivative_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::bessel_i_scaled_derivative(
            order.real_component(),
            z.real_component(),
            degree,
        )
        .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_i_complex_scaled_derivative(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::bessel_i_scaled_derivative_evaluate(
            order.real_component(),
            z.real_component(),
            degree,
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_i_complex_scaled_derivative_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
            options,
            context,
        )
    }
    scoped pub bessel_i_scaled_derivative_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `e^z K_ν⁽ᵏ⁾(z)`, the scaled `k`-th argument derivative of [`bessel_k`](crate::bessel_k), where `k` is `degree`.

```text
K_ν⁽ᵏ⁾(z) = (−1)^k 2^(−k) Σₘ₌₀^k C(k, m) K_(ν−k+2m)(z)      (DLMF §10.29)
```

`C(k, m)` is the binomial coefficient and the order `ν` is held fixed; `k = 0` returns the
function itself. Orders, domain and branch cut are those of
[`bessel_k_scaled`](crate::bessel_k_scaled). Each term is computed with the same kernels, so the
cost grows linearly with `k`.

The result is `e^z` times the `k`-th derivative of the unscaled function; the factor itself is
not differentiated.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_special::{bessel_k, bessel_k_scaled_derivative};

// eˣ K₀′(x) = −eˣ K₁(x)
let expected = -(1.0_f64.exp()) * bessel_k(1.0_f64, 1.0)?;
assert!((bessel_k_scaled_derivative(0.0_f64, 1.0, 1)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Derivatives have no uniform ulp bound. The detailed form reports an error estimate
that includes the rounding of the binomial sum. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`, also for `k = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `ConvergenceFailure` (`IterationLimit`) if `k` exceeds the remaining iteration budget.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub bessel_k_scaled_derivative [T: NumericScalar] (order: impl crate::Parameter<T>, z: T, degree: u32) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_k_scaled_derivative_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::bessel_k_scaled_derivative(
            order.real_component(),
            z.real_component(),
            degree,
        )
        .map(T::from_real)
    } complex {
        crate::bessel_airy::complex_bessel::bessel_k_complex_scaled_derivative(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::bessel_k_scaled_derivative_evaluate(
            order.real_component(),
            z.real_component(),
            degree,
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::bessel_k_complex_scaled_derivative_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
            options,
            context,
        )
    }
    scoped pub bessel_k_scaled_derivative_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `k`-th argument derivative of [`hankel_1`](crate::hankel_1), where `k` is `degree`.

```text
H⁽¹⁾_ν⁽ᵏ⁾(z) = 2^(−k) Σₘ₌₀^k (−1)^m C(k, m) H⁽¹⁾_(ν−k+2m)(z)      (DLMF §10.6)
```

`C(k, m)` is the binomial coefficient and the order `ν` is held fixed; `k = 0` returns the
function itself. Orders, domain and branch cut are those of [`hankel_1`](crate::hankel_1). Each
term is computed with the same kernels, so the cost grows linearly with `k`.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_special::{hankel_1, hankel_1_derivative};

// H⁽¹⁾₀′(x) = −H⁽¹⁾₁(x) (DLMF §10.6)
let expected = -hankel_1(1.0_f64, 1.0)?;
assert!((hankel_1_derivative(0.0_f64, 1.0, 1)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Derivatives have no uniform ulp bound. The detailed form reports an error estimate
that includes the rounding of the binomial sum. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`, also for `k = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `ConvergenceFailure` (`IterationLimit`) if `k` exceeds the remaining iteration budget.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub hankel_1_derivative [T: NumericScalar] (order: impl crate::Parameter<T>, z: T, degree: u32) -> Complex<T::Real>
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed hankel_1_derivative_with_options -> SpecialFunctionResult<Complex<T::Real>>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::hankel_1_derivative(
            order.real_component(),
            z.real_component(),
            degree,
        )
    } complex {
        crate::bessel_airy::complex_bessel::hankel_1_complex_derivative(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
        )
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::hankel_1_derivative_evaluate(
            order.real_component(),
            z.real_component(),
            degree,
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::hankel_1_complex_derivative_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
            options,
            context,
        )
    }
    scoped pub hankel_1_derivative_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `k`-th argument derivative of [`hankel_2`](crate::hankel_2), where `k` is `degree`.

```text
H⁽²⁾_ν⁽ᵏ⁾(z) = 2^(−k) Σₘ₌₀^k (−1)^m C(k, m) H⁽²⁾_(ν−k+2m)(z)      (DLMF §10.6)
```

`C(k, m)` is the binomial coefficient and the order `ν` is held fixed; `k = 0` returns the
function itself. Orders, domain and branch cut are those of [`hankel_2`](crate::hankel_2). Each
term is computed with the same kernels, so the cost grows linearly with `k`.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_special::{hankel_2, hankel_2_derivative};

// H⁽²⁾₀′(x) = −H⁽²⁾₁(x) (DLMF §10.6)
let expected = -hankel_2(1.0_f64, 1.0)?;
assert!((hankel_2_derivative(0.0_f64, 1.0, 1)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Derivatives have no uniform ulp bound. The detailed form reports an error estimate
that includes the rounding of the binomial sum. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`, also for `k = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.
- `ConvergenceFailure` (`IterationLimit`) if `k` exceeds the remaining iteration budget.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    #[inline]
    pub hankel_2_derivative [T: NumericScalar] (order: impl crate::Parameter<T>, z: T, degree: u32) -> Complex<T::Real>
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed hankel_2_derivative_with_options -> SpecialFunctionResult<Complex<T::Real>>;
    dispatch T;
    value real {
        crate::bessel_airy::complex_bessel::hankel_2_derivative(
            order.real_component(),
            z.real_component(),
            degree,
        )
    } complex {
        crate::bessel_airy::complex_bessel::hankel_2_complex_derivative(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
        )
    }
    evaluate |options, context| real {
        crate::bessel_airy::complex_bessel::hankel_2_derivative_evaluate(
            order.real_component(),
            z.real_component(),
            degree,
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::hankel_2_complex_derivative_evaluation(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            degree,
            options,
            context,
        )
    }
    scoped pub hankel_2_derivative_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Airy function of the first kind `Ai(z)`.

```text
w″(z) = z w(z)      (DLMF 9.2.1)
```

`Ai` and `Bi` are the solutions of the Airy equation with `Ai(0) = 3^(−2/3)/Γ(2/3)`,
`Ai′(0) = −3^(−1/3)/Γ(1/3)`, `Bi(0) = 3^(−1/6)/Γ(2/3)` and `Bi′(0) = 3^(1/6)/Γ(1/3)` (DLMF
§9.2). Both are entire, so every finite real or complex argument is accepted and there is no
branch cut. `Ai(x)` decays like `e^(−(2/3) x^(3/2))` as `x → +∞` and oscillates as `x → −∞`; for
large positive `x` the result underflows to zero.

For real `|x| ≤ 16` the function is a Taylor polynomial of degree 35 about the nearest integer,
generated from the differential equation and evaluated with compensated arithmetic where
cancellation requires it. Larger `|x|` uses the asymptotic expansions of DLMF §9.7, with the
phase reduced exactly on the negative axis. Complex arguments combine Taylor series about real
points with the same asymptotic expansions. The work is bounded by the iteration budget of
[`AiryOptions`](crate::AiryOptions). [`airy_ai_with_options`](crate::airy_ai_with_options) also
returns `Ai′(z)` when it is representable.

# Examples

```
// Ai(0) = 3^(−2/3)/Γ(2/3) (DLMF §9.2)
let ai = scioxide_special::airy_ai(0.0_f64)?;
assert!((ai - 0.3550280538878172).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `x`, the error is within 64 ulps for `f64` and `f32` on the tested range;
near a zero of the function the tests bound the absolute error instead. Complex arguments have
no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|Ai(z)|` exceeds the range of the scalar
  type, which happens only for complex `z` of large modulus with `π/3 < |arg z| < π`;
  [`airy_ai_scaled`](crate::airy_ai_scaled) avoids this.

# References

- NIST DLMF, Chapter 9. <https://dlmf.nist.gov/9>"###)]
    #[inline]
    pub airy_ai [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::AiryOptions;
    detailed airy_ai_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::airy::airy_ai(x.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_airy::airy_ai_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::airy::airy_ai_evaluation(x.real_component(), options, context)
    } complex {
        crate::bessel_airy::complex_airy::airy_ai_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub airy_ai_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Airy function of the second kind `Bi(z)`.

```text
w″(z) = z w(z)      (DLMF 9.2.1)
```

`Ai` and `Bi` are the solutions of the Airy equation with `Ai(0) = 3^(−2/3)/Γ(2/3)`,
`Ai′(0) = −3^(−1/3)/Γ(1/3)`, `Bi(0) = 3^(−1/6)/Γ(2/3)` and `Bi′(0) = 3^(1/6)/Γ(1/3)` (DLMF
§9.2). Both are entire, so every finite real or complex argument is accepted and there is no
branch cut. `Bi(x)` grows like `e^((2/3) x^(3/2))` as `x → +∞` and oscillates as `x → −∞`.

For real `|x| ≤ 16` the function is a Taylor polynomial of degree 35 about the nearest integer,
generated from the differential equation and evaluated with compensated arithmetic where
cancellation requires it. Larger `|x|` uses the asymptotic expansions of DLMF §9.7, with the
phase reduced exactly on the negative axis. Complex arguments combine Taylor series about real
points with the same asymptotic expansions. The work is bounded by the iteration budget of
[`AiryOptions`](crate::AiryOptions). [`airy_bi_with_options`](crate::airy_bi_with_options) also
returns `Bi′(z)` when it is representable.

# Examples

```
// Bi(0) = 3^(−1/6)/Γ(2/3) (DLMF §9.2)
let bi = scioxide_special::airy_bi(0.0_f64)?;
assert!((bi - 0.6149266274460007).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `x`, the error is within 64 ulps for `f64` and `f32` on the tested range;
near a zero of the function the tests bound the absolute error instead. Complex arguments have
no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|Bi(z)|` exceeds the range of the scalar
  type, as for real `x` above about 104 in `f64`; [`airy_bi_scaled`](crate::airy_bi_scaled)
  avoids this.

# References

- NIST DLMF, Chapter 9. <https://dlmf.nist.gov/9>"###)]
    #[inline]
    pub airy_bi [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::AiryOptions;
    detailed airy_bi_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::airy::airy_bi(x.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_airy::airy_bi_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::airy::airy_bi_evaluation(x.real_component(), options, context)
    } complex {
        crate::bessel_airy::complex_airy::airy_bi_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub airy_bi_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Airy function `e^ζ Ai(z)`, where `ζ = (2/3) z^(3/2)`.

For real `x > 0` the result is `e^((2/3) x^(3/2)) Ai(x)`; for real `x ≤ 0` it is `Ai(x)` without
a factor. For complex `z` the factor is `e^ζ` with `ζ = (2/3) z √z` and the principal square
root, applied everywhere. On the negative real axis `e^ζ` is a unit-modulus phase, so the
complex result there differs from the real one, and the sign of the zero imaginary part selects
the side. The scaled value stays in range wherever `Ai` grows or decays exponentially. The
methods are those of [`airy_ai`](crate::airy_ai).
[`airy_ai_scaled_with_options`](crate::airy_ai_scaled_with_options) also returns the scaled
derivative `e^ζ Ai′(z)` when it is representable; the factor itself is not differentiated.

# Examples

```
// e^ζ Ai(x) with ζ = (2/3) x^(3/2)
let x = 2.0_f64;
let zeta = 2.0 / 3.0 * x * x.sqrt();
let expected = zeta.exp() * scioxide_special::airy_ai(x)?;
assert!((scioxide_special::airy_ai_scaled(x)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `x`, the error is within 64 ulps for `f64` and `f32` on the tested range;
near a zero of the function the tests bound the absolute error instead. Complex arguments have
no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"No conditions beyond those above.

# References

- NIST DLMF, Chapter 9. <https://dlmf.nist.gov/9>"###)]
    #[inline]
    pub airy_ai_scaled [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::AiryOptions;
    detailed airy_ai_scaled_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::airy::airy_ai_scaled(x.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_airy::airy_ai_complex_scaled(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::airy::airy_ai_scaled_evaluation(x.real_component(), options, context)
    } complex {
        crate::bessel_airy::complex_airy::airy_ai_complex_scaled_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub airy_ai_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Airy function `e^(−|Re ζ|) Bi(z)`, where `ζ = (2/3) z^(3/2)`.

For real `x > 0` the result is `e^(−(2/3) x^(3/2)) Bi(x)`; for real `x ≤ 0` it is `Bi(x)`
without a factor. For complex `z` the factor is `e^(−|Re ζ|)` with `ζ = (2/3) z √z` and the
principal square root. The scaled value stays in range where `Bi` grows exponentially. The
methods are those of [`airy_bi`](crate::airy_bi).
[`airy_bi_scaled_with_options`](crate::airy_bi_scaled_with_options) also returns the scaled
derivative `e^(−|Re ζ|) Bi′(z)` when it is representable; the factor itself is not differentiated.

# Examples

```
// e^(−ζ) Bi(x) with ζ = (2/3) x^(3/2)
let x = 2.0_f64;
let zeta = 2.0 / 3.0 * x * x.sqrt();
let expected = (-zeta).exp() * scioxide_special::airy_bi(x)?;
assert!((scioxide_special::airy_bi_scaled(x)? - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `x`, the error is within 64 ulps for `f64` and `f32` on the tested range;
near a zero of the function the tests bound the absolute error instead. Complex arguments have
no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"No conditions beyond those above.

# References

- NIST DLMF, Chapter 9. <https://dlmf.nist.gov/9>"###)]
    #[inline]
    pub airy_bi_scaled [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::AiryOptions;
    detailed airy_bi_scaled_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::airy::airy_bi_scaled(x.real_component()).map(T::from_real)
    } complex {
        crate::bessel_airy::complex_airy::airy_bi_complex_scaled(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::airy::airy_bi_scaled_evaluation(x.real_component(), options, context)
    } complex {
        crate::bessel_airy::complex_airy::airy_bi_complex_scaled_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub airy_bi_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Bessel function `e^(−|Im z|) J_ν(z)`.

For real arguments the factor is 1 and the result equals [`bessel_j`](crate::bessel_j). For
complex `z` the scaled value stays in range for large `|Im z|`, where `J_ν(z)` grows like
`e^|Im z|`. Orders, domain, branch cut and origin values are those of
[`bessel_j`](crate::bessel_j). In the detailed result the derivative is `e^(−|Im z|) J_ν′(z)`;
the factor itself is not differentiated.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_core::Complex;
use scioxide_special::{bessel_j, bessel_j_scaled};

// e^(−|Im z|) J₀(z) at z = 1 + 2i
let (nu, z) = (Complex::new(0.0_f64, 0.0), Complex::new(1.0, 2.0));
let expected = bessel_j(nu, z)?.scale((-2.0_f64).exp());
assert!((bessel_j_scaled(nu, z)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the accuracy is that of [`bessel_j`](crate::bessel_j). Complex
arguments have no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for real `x < 0` when `ν` is not an integer.
- `DomainError` (`Pole`) at `z = 0` when `ν` is a negative non-integer, or complex with
  `Re ν ≤ 0`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    pub bessel_j_scaled [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_j_scaled_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::bessel_j(order.real_component(), z.real_component())
            .map(T::from_real)
    } complex {
        let options = crate::BesselOptions::default();
        let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
        let _ = &context;
        crate::bessel_airy::complex_bessel::evaluate_controlled(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            crate::bessel_airy::complex_bessel::Kind::J,
            true,
            scioxide_core::algorithm_component!("bessel_j_scaled"),
            context,
        )?
        .value()
        .map(crate::dispatch::convert_scalar::<_, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::bessel_j_evaluate(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::evaluate_controlled(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            crate::bessel_airy::complex_bessel::Kind::J,
            true,
            scioxide_core::algorithm_component!("bessel_j_scaled"),
            context,
        )
    }
    scoped pub bessel_j_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Bessel function `e^(−|Im z|) Y_ν(z)`.

For real arguments the factor is 1 and the result equals [`bessel_y`](crate::bessel_y). For
complex `z` the scaled value stays in range for large `|Im z|`, where `Y_ν(z)` grows like
`e^|Im z|`. Orders, domain, branch cut and origin values are those of
[`bessel_y`](crate::bessel_y). In the detailed result the derivative is `e^(−|Im z|) Y_ν′(z)`;
the factor itself is not differentiated.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_core::Complex;
use scioxide_special::{bessel_y, bessel_y_scaled};

// e^(−|Im z|) Y₀(z) at z = 1 + 2i
let (nu, z) = (Complex::new(0.0_f64, 0.0), Complex::new(1.0, 2.0));
let expected = bessel_y(nu, z)?.scale((-2.0_f64).exp());
assert!((bessel_y_scaled(nu, z)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the accuracy is that of [`bessel_y`](crate::bessel_y). Complex
arguments have no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]
    pub bessel_y_scaled [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [order] to T;
    options options: crate::BesselOptions;
    detailed bessel_y_scaled_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::bessel_airy::real_api::bessel_y(order.real_component(), z.real_component())
            .map(T::from_real)
    } complex {
        let options = crate::BesselOptions::default();
        let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
        let _ = &context;
        crate::bessel_airy::complex_bessel::evaluate_controlled(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            crate::bessel_airy::complex_bessel::Kind::Y,
            true,
            scioxide_core::algorithm_component!("bessel_y_scaled"),
            context,
        )?
        .value()
        .map(crate::dispatch::convert_scalar::<_, T>)
    }
    evaluate |options, context| real {
        crate::bessel_airy::real_api::bessel_y_evaluate(
            order.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::bessel_airy::complex_bessel::evaluate_controlled(
            crate::dispatch::complex(order),
            crate::dispatch::complex(z),
            options,
            crate::bessel_airy::complex_bessel::Kind::Y,
            true,
            scioxide_core::algorithm_component!("bessel_y_scaled"),
            context,
        )
    }
    scoped pub bessel_y_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {

    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Hankel function `e^(−iz) H⁽¹⁾_ν(z)`.

For real `x > 0` the factor is a unit-modulus phase. For complex `z` the scaled value stays in
range for large negative `Im z`, where `H⁽¹⁾_ν(z)` grows like `e^|Im z|`. Orders, domain and
branch cut are those of [`hankel_1`](crate::hankel_1). In the detailed result the derivative is
`e^(−iz) H⁽¹⁾_ν′(z)`; the factor itself is not differentiated.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_core::Complex;
use scioxide_special::{hankel_1, hankel_1_scaled};

// e^(−iz) H⁽¹⁾₀(x) at x = 1
let expected = hankel_1(0.0_f64, 1.0)? * Complex::new(1.0_f64.cos(), -1.0_f64.sin());
assert!((hankel_1_scaled(0.0_f64, 1.0)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]


pub hankel_1_scaled [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> Complex<T::Real>
where [];
promote [order] to T;
options options: crate::BesselOptions;
detailed hankel_1_scaled_with_options -> SpecialFunctionResult<Complex<T::Real>>;
evaluate |options, context| {
    if T::IS_REAL {
        crate::support::check_positive(z.real_component(), scioxide_core::argument!("z"),
            "real Hankel argument must be positive")?;
    }
 crate::bessel_airy::complex_bessel::evaluate_controlled(crate::dispatch::complex(order), crate::dispatch::complex(z), options, crate::bessel_airy::complex_bessel::Kind::H1, true, scioxide_core::algorithm_component!("hankel_1_scaled"), context) }
scoped pub hankel_1_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {

    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Hankel function `e^(iz) H⁽²⁾_ν(z)`.

For real `x > 0` the factor is a unit-modulus phase. For complex `z` the scaled value stays in
range for large positive `Im z`, where `H⁽²⁾_ν(z)` grows like `e^|Im z|`. Orders, domain and
branch cut are those of [`hankel_2`](crate::hankel_2). In the detailed result the derivative is
`e^(iz) H⁽²⁾_ν′(z)`; the factor itself is not differentiated.

`ν` may be real while `z` is complex.

# Examples

```
use scioxide_core::Complex;
use scioxide_special::{hankel_2, hankel_2_scaled};

// e^(iz) H⁽²⁾₀(x) at x = 1
let expected = hankel_2(0.0_f64, 1.0)? * Complex::new(1.0_f64.cos(), 1.0_f64.sin());
assert!((hankel_2_scaled(0.0_f64, 1.0)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.
- `DomainError` (`OutsideDomain`) for real `x < 0`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]


pub hankel_2_scaled [T: NumericScalar] (order: impl crate::Parameter<T>, z: T) -> Complex<T::Real>
where [];
promote [order] to T;
options options: crate::BesselOptions;
detailed hankel_2_scaled_with_options -> SpecialFunctionResult<Complex<T::Real>>;
evaluate |options, context| {
    if T::IS_REAL {
        crate::support::check_positive(z.real_component(), scioxide_core::argument!("z"),
            "real Hankel argument must be positive")?;
    }
 crate::bessel_airy::complex_bessel::evaluate_controlled(crate::dispatch::complex(order), crate::dispatch::complex(z), options, crate::bessel_airy::complex_bessel::Kind::H2, true, scioxide_core::algorithm_component!("hankel_2_scaled"), context) }
scoped pub hankel_2_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {

    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled spherical Hankel function `e^(−iz) h⁽¹⁾_n(z)`.

For real `x` the factor is a unit-modulus phase. For complex `z` the scaled value stays in range
for large negative `Im z`. Orders and domain are those of
[`spherical_hankel_1`](crate::spherical_hankel_1). In the detailed result the derivative is
`e^(−iz) h⁽¹⁾_n′(z)`; the factor itself is not differentiated.

# Examples

```
use scioxide_core::Complex;
use scioxide_special::{spherical_hankel_1, spherical_hankel_1_scaled};

// e^(−iz) h⁽¹⁾₀(x) at x = 1
let expected = spherical_hankel_1(0, 1.0_f64)? * Complex::new(1.0_f64.cos(), -1.0_f64.sin());
assert!((spherical_hankel_1_scaled(0, 1.0_f64)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]


pub spherical_hankel_1_scaled [T: NumericScalar] (order: u32, z: T) -> Complex<T::Real> where [];
options options: crate::SphericalBesselOptions;
detailed spherical_hankel_1_scaled_with_options -> SpecialFunctionResult<Complex<T::Real>>;
evaluate |options, context| { crate::bessel_airy::complex_bessel::spherical_evaluate_controlled(order, crate::dispatch::complex(z), options, crate::bessel_airy::complex_bessel::Kind::H1, true, scioxide_core::algorithm_component!("spherical_hankel_1_scaled"), context) }
scoped pub spherical_hankel_1_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {

    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled spherical Hankel function `e^(iz) h⁽²⁾_n(z)`.

For real `x` the factor is a unit-modulus phase. For complex `z` the scaled value stays in range
for large positive `Im z`. Orders and domain are those of
[`spherical_hankel_2`](crate::spherical_hankel_2). In the detailed result the derivative is
`e^(iz) h⁽²⁾_n′(z)`; the factor itself is not differentiated.

# Examples

```
use scioxide_core::Complex;
use scioxide_special::{spherical_hankel_2, spherical_hankel_2_scaled};

// e^(iz) h⁽²⁾₀(x) at x = 1
let expected = spherical_hankel_2(0, 1.0_f64)? * Complex::new(1.0_f64.cos(), 1.0_f64.sin());
assert!((spherical_hankel_2_scaled(0, 1.0_f64)? - expected).norm() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Spherical Bessel functions have no uniform ulp bound. Tests compare real and complex
values with independent references, using a mixed absolute and relative tolerance near zeros.
The detailed form reports an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0`.

# References

- NIST DLMF, §10.47–10.60. <https://dlmf.nist.gov/10.47>"###)]


pub spherical_hankel_2_scaled [T: NumericScalar] (order: u32, z: T) -> Complex<T::Real> where [];
options options: crate::SphericalBesselOptions;
detailed spherical_hankel_2_scaled_with_options -> SpecialFunctionResult<Complex<T::Real>>;
evaluate |options, context| { crate::bessel_airy::complex_bessel::spherical_evaluate_controlled(order, crate::dispatch::complex(z), options, crate::bessel_airy::complex_bessel::Kind::H2, true, scioxide_core::algorithm_component!("spherical_hankel_2_scaled"), context) }
scoped pub spherical_hankel_2_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
