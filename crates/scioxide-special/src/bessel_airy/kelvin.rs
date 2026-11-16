//! Kelvin functions from I_0 and K_0 on the pi/4 ray (DLMF 10.61).
//! Ber/bei share the order-zero real series through |x|=10 (DLMF 10.65).
//! Real paired logarithmic series below 20 and scaled asymptotic sums above it
//! cover the ordinary range (DLMF 10.31 and 10.67). Extreme arguments use
//! the scaled Bessel kernels. Values and analytic derivatives share work.

use crate::complex_support::ScaledForm;
use crate::result::{Bound, Either, Evaluation, Form};
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, real};
use crate::{KelvinOptions, SpecialFunctionResult};
use scioxide_core::algorithm_component;
use scioxide_core::{RealScalar, SciOxideError};

type Detailed<S> = SpecialFunctionResult<S>;
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Kelvin function `ber(x)` of order zero.

```text
ber(x) + i bei(x) = I₀(x e^(iπ/4)) = J₀(x e^(3iπ/4)),
ker(x) + i kei(x) = K₀(x e^(iπ/4)),
```

where `I₀`, `J₀` and `K₀` are the Bessel functions of order zero (DLMF 10.61.1–2).

`ber` is an even entire function. Every finite real `x` is accepted, and `ber(0) = 1`. Use
[`ber_scaled`](crate::ber_scaled) for large `|x|` and [`kelvin`](crate::kelvin) for all four
Kelvin functions at once.

For `|x| ≤ 10` the function sums its power series (DLMF 10.65) in double-word arithmetic.
Larger arguments use `I₀(|x| e^(iπ/4))`, from a power series below 20 and the asymptotic
expansion (DLMF 10.67) from 20.

# Examples

```
// ber(0) = 1 (DLMF 10.65)
let b = scioxide_special::ber(0.0_f64)?;
assert!((b - 1.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`ber_with_options`](crate::ber_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|ber(x)|` exceeds the range of the scalar type;
  [`ber_scaled`](crate::ber_scaled) stays finite.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub ber [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed ber_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (0, false), false, context)
    }
    scoped pub ber_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Kelvin function `bei(x)` of order zero.

```text
ber(x) + i bei(x) = I₀(x e^(iπ/4)) = J₀(x e^(3iπ/4)),
ker(x) + i kei(x) = K₀(x e^(iπ/4)),
```

where `I₀`, `J₀` and `K₀` are the Bessel functions of order zero (DLMF 10.61.1–2).

`bei` is an even entire function. Every finite real `x` is accepted, and `bei(0) = 0`. Use
[`bei_scaled`](crate::bei_scaled) for large `|x|` and [`kelvin`](crate::kelvin) for all four
Kelvin functions at once.

For `|x| ≤ 10` the function sums its power series (DLMF 10.65) in double-word arithmetic.
Larger arguments use `I₀(|x| e^(iπ/4))`, from a power series below 20 and the asymptotic
expansion (DLMF 10.67) from 20.

# Examples

```
// bei(1) = 0.2495660400366597…
let b = scioxide_special::bei(1.0_f64)?;
assert!((b - 0.249_566_040_036_659_72).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`bei_with_options`](crate::bei_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|bei(x)|` exceeds the range of the scalar type;
  [`bei_scaled`](crate::bei_scaled) stays finite.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub bei [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed bei_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (1, false), false, context)
    }
    scoped pub bei_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Kelvin function `ker(x)` of order zero.

```text
ber(x) + i bei(x) = I₀(x e^(iπ/4)) = J₀(x e^(3iπ/4)),
ker(x) + i kei(x) = K₀(x e^(iπ/4)),
```

where `I₀`, `J₀` and `K₀` are the Bessel functions of order zero (DLMF 10.61.1–2).

`ker` is defined for `x > 0` and has a logarithmic singularity at `x = 0`. It decays like
`exp(−x/√2)`; use [`ker_scaled`](crate::ker_scaled) for large `x`.

The function uses `K₀(x e^(iπ/4))`, from a power series in double-word arithmetic below
`x = 20` and the asymptotic expansion (DLMF 10.67) from 20.

# Examples

```
// ker(1) = 0.2867062087283160…
let k = scioxide_special::ker(1.0_f64)?;
assert!((k - 0.286_706_208_728_316_04).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`ker_with_options`](crate::ker_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`, where `ker(x) ~ −ln(x/2) − γ` (Euler's constant `γ`).
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub ker [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed ker_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (2, false), false, context)
    }
    scoped pub ker_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Kelvin function `kei(x)` of order zero.

```text
ber(x) + i bei(x) = I₀(x e^(iπ/4)) = J₀(x e^(3iπ/4)),
ker(x) + i kei(x) = K₀(x e^(iπ/4)),
```

where `I₀`, `J₀` and `K₀` are the Bessel functions of order zero (DLMF 10.61.1–2).

`kei` is defined for `x ≥ 0`, with the continuous value `kei(0) = −π/4`. It decays like
`exp(−x/√2)`; use [`kei_scaled`](crate::kei_scaled) for large `x`.

The function uses `K₀(x e^(iπ/4))`, from a power series in double-word arithmetic below
`x = 20` and the asymptotic expansion (DLMF 10.67) from 20.

# Examples

```
// kei(0) = −π/4 (DLMF 10.65)
let k = scioxide_special::kei(0.0_f64)?;
assert!((k + core::f64::consts::FRAC_PI_4).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`kei_with_options`](crate::kei_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub kei [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed kei_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (3, false), false, context)
    }
    scoped pub kei_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the derivative `ber′(x)` of the Kelvin function [`ber`](crate::ber).

`ber′` is an odd entire function. Every finite real `x` is accepted, and `ber′(0) = 0`.
The value comes from the same expansions as [`ber`](crate::ber).

# Examples

```
// ber′(1) = −0.0624457521790309…
let d = scioxide_special::berp(1.0_f64)?;
assert!((d + 0.062_445_752_179_030_96).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`berp_with_options`](crate::berp_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|ber′(x)|` exceeds the range of the scalar type;
  [`berp_scaled`](crate::berp_scaled) stays finite.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub berp [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed berp_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (0, true), false, context)
    }
    scoped pub berp_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the derivative `bei′(x)` of the Kelvin function [`bei`](crate::bei).

`bei′` is an odd entire function. Every finite real `x` is accepted, and `bei′(0) = 0`.
The value comes from the same expansions as [`bei`](crate::bei).

# Examples

```
// bei′(1) = 0.4973965114680973…
let d = scioxide_special::beip(1.0_f64)?;
assert!((d - 0.497_396_511_468_097_3).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`beip_with_options`](crate::beip_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|bei′(x)|` exceeds the range of the scalar type;
  [`beip_scaled`](crate::beip_scaled) stays finite.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub beip [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed beip_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (1, true), false, context)
    }
    scoped pub beip_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the derivative `ker′(x)` of the Kelvin function [`ker`](crate::ker).

`ker′` is defined for `x > 0` and behaves like `−1/x` at the origin. The value comes from the
same expansions as [`ker`](crate::ker).

# Examples

```
// ker′(1) = −0.6946038911006905…
let d = scioxide_special::kerp(1.0_f64)?;
assert!((d + 0.694_603_891_100_690_5).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`kerp_with_options`](crate::kerp_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`, where `ker′(x) ~ −1/x`.
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub kerp [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed kerp_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (2, true), false, context)
    }
    scoped pub kerp_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the derivative `kei′(x)` of the Kelvin function [`kei`](crate::kei).

`kei′` is defined for `x ≥ 0`, with the continuous value `kei′(0) = 0`. The value comes from
the same expansions as [`kei`](crate::kei).

# Examples

```
// kei′(1) = 0.3523699133361705…
let d = scioxide_special::keip(1.0_f64)?;
assert!((d - 0.352_369_913_336_170_5).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`keip_with_options`](crate::keip_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub keip [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed keip_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (3, true), false, context)
    }
    scoped pub keip_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Kelvin function `ber(x) exp(−|x|/√2)`.

`ber(x)` grows like `exp(|x|/√2) / √(2π|x|)` (DLMF 10.67), so the scaled value stays finite
where [`ber`](crate::ber) overflows.
The domain, parity and origin value are those of [`ber`](crate::ber).

# Examples

```
// ber_scaled(x) = ber(x) exp(−|x|/√2)
let x = 2.0_f64;
let expected = scioxide_special::ber(x)? * (-x / 2.0_f64.sqrt()).exp();
assert!((scioxide_special::ber_scaled(x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`ber_scaled_with_options`](crate::ber_scaled_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"Every finite `x` is accepted; only the general conditions above apply.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub ber_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed ber_scaled_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (0, false), true, context)
    }
    scoped pub ber_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Kelvin function `bei(x) exp(−|x|/√2)`.

`bei(x)` grows like `exp(|x|/√2) / √(2π|x|)` (DLMF 10.67), so the scaled value stays finite
where [`bei`](crate::bei) overflows.
The domain, parity and origin value are those of [`bei`](crate::bei).

# Examples

```
// bei_scaled(x) = bei(x) exp(−|x|/√2)
let x = 2.0_f64;
let expected = scioxide_special::bei(x)? * (-x / 2.0_f64.sqrt()).exp();
assert!((scioxide_special::bei_scaled(x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`bei_scaled_with_options`](crate::bei_scaled_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"Every finite `x` is accepted; only the general conditions above apply.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub bei_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed bei_scaled_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (1, false), true, context)
    }
    scoped pub bei_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Kelvin function `ker(x) exp(x/√2)`.

`ker(x)` decays like `exp(−x/√2) √(π/(2x))` (DLMF 10.67), so the scaled value does not
underflow for large `x`.
The domain and origin behavior are those of [`ker`](crate::ker).

# Examples

```
// ker_scaled(x) = ker(x) exp(x/√2)
let x = 2.0_f64;
let expected = scioxide_special::ker(x)? * (x / 2.0_f64.sqrt()).exp();
assert!((scioxide_special::ker_scaled(x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`ker_scaled_with_options`](crate::ker_scaled_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`, where `ker(x) ~ −ln(x/2) − γ` (Euler's constant `γ`).
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub ker_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed ker_scaled_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (2, false), true, context)
    }
    scoped pub ker_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponentially scaled Kelvin function `kei(x) exp(x/√2)`.

`kei(x)` decays like `exp(−x/√2) √(π/(2x))` (DLMF 10.67), so the scaled value does not
underflow for large `x`.
The domain and origin behavior are those of [`kei`](crate::kei).

# Examples

```
// kei_scaled(x) = kei(x) exp(x/√2)
let x = 2.0_f64;
let expected = scioxide_special::kei(x)? * (x / 2.0_f64.sqrt()).exp();
assert!((scioxide_special::kei_scaled(x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`kei_scaled_with_options`](crate::kei_scaled_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub kei_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed kei_scaled_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (3, false), true, context)
    }
    scoped pub kei_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ber′(x) exp(−|x|/√2)`, the derivative of ber with the scale factor of
[`ber_scaled`](crate::ber_scaled).

The factor multiplies `ber′(x)`; the result is not the derivative of `ber_scaled`.
The domain, parity and origin value are those of [`berp`](crate::berp).

# Examples

```
// berp_scaled(x) = ber′(x) exp(−|x|/√2)
let x = 2.0_f64;
let expected = scioxide_special::berp(x)? * (-x / 2.0_f64.sqrt()).exp();
assert!((scioxide_special::berp_scaled(x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`berp_scaled_with_options`](crate::berp_scaled_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"Every finite `x` is accepted; only the general conditions above apply.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub berp_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed berp_scaled_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (0, true), true, context)
    }
    scoped pub berp_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `bei′(x) exp(−|x|/√2)`, the derivative of bei with the scale factor of
[`bei_scaled`](crate::bei_scaled).

The factor multiplies `bei′(x)`; the result is not the derivative of `bei_scaled`.
The domain, parity and origin value are those of [`beip`](crate::beip).

# Examples

```
// beip_scaled(x) = bei′(x) exp(−|x|/√2)
let x = 2.0_f64;
let expected = scioxide_special::beip(x)? * (-x / 2.0_f64.sqrt()).exp();
assert!((scioxide_special::beip_scaled(x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`beip_scaled_with_options`](crate::beip_scaled_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"Every finite `x` is accepted; only the general conditions above apply.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub beip_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed beip_scaled_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (1, true), true, context)
    }
    scoped pub beip_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ker′(x) exp(x/√2)`, the derivative of ker with the scale factor of
[`ker_scaled`](crate::ker_scaled).

The factor multiplies `ker′(x)`; the result is not the derivative of `ker_scaled`.
The domain and origin behavior are those of [`kerp`](crate::kerp).

# Examples

```
// kerp_scaled(x) = ker′(x) exp(x/√2)
let x = 2.0_f64;
let expected = scioxide_special::kerp(x)? * (x / 2.0_f64.sqrt()).exp();
assert!((scioxide_special::kerp_scaled(x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`kerp_scaled_with_options`](crate::kerp_scaled_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`, where `ker′(x) ~ −1/x`.
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub kerp_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed kerp_scaled_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (2, true), true, context)
    }
    scoped pub kerp_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `kei′(x) exp(x/√2)`, the derivative of kei with the scale factor of
[`kei_scaled`](crate::kei_scaled).

The factor multiplies `kei′(x)`; the result is not the derivative of `kei_scaled`.
The domain and origin behavior are those of [`keip`](crate::keip).

# Examples

```
// keip_scaled(x) = kei′(x) exp(x/√2)
let x = 2.0_f64;
let expected = scioxide_special::keip(x)? * (x / 2.0_f64.sqrt()).exp();
assert!((scioxide_special::keip_scaled(x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`keip_scaled_with_options`](crate::keip_scaled_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub keip_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: KelvinOptions;
    detailed keip_scaled_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate_controlled(x, options, (3, true), true, context)
    }
    scoped pub keip_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

fn evaluate_controlled<S: RealScalar>(
    x: S,
    options: KelvinOptions,
    (index, derivative): (usize, bool),
    scaled: bool,
    parent: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<KelvinEvaluation<S>, SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "finite argument required",
        ));
    }
    if index >= 2 && (x < S::ZERO || index == 2 && x == S::ZERO) {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "ker and its derivative are singular at zero; recessive Kelvin functions require nonnegative x",
        ));
    }
    let a = x.abs().try_to_f64_exact()?;
    let mut work = Work::with_batched_context(options.iteration_budget().get(), parent)?;
    let estimate = if a == 0.0 {
        origin(index, derivative, &mut work)
    } else {
        pair_estimates(a, index >= 2, scaled, &mut work)
            .map(|pair| select(pair, index % 2, derivative, x.is_sign_negative(), a))
    };
    work.check()?;
    report(estimate, work.used, work.exhausted())
}

fn report<S: RealScalar>(
    estimate: Result<Estimate, SciOxideError>,
    used: usize,
    exhausted: bool,
) -> Result<KelvinEvaluation<S>, SciOxideError> {
    if !exhausted && let Ok(e) = estimate.as_ref() {
        let value = e.y.v.re;
        if e.scale.v == real(0.0)
            && e.error.is_finite()
            && value.abs() >= S::MIN_POSITIVE.into_f64()
            && S::try_from_f64_lossy(value).is_ok()
        {
            let projected = S::try_from_f64_lossy(value)?;
            let derivative = e.y.d.re;
            let derivative_error = e.derivative_error.unwrap_or(e.error);
            let slope = if derivative.is_finite()
                && derivative_error.is_finite()
                && S::try_from_f64_lossy(derivative).is_ok()
            {
                let projected = S::try_from_f64_lossy(derivative)?;
                (projected != S::ZERO || derivative == 0.0).then_some((
                    projected,
                    Bound::derived(crate::result::outward::<S>(
                        derivative_error
                            + (projected.into_f64() - derivative).abs()
                            + 24.0 * f64::EPSILON * derivative.abs(),
                    )?),
                ))
            } else {
                None
            };
            return Ok(Evaluation::finite(
                projected,
                Bound::derived(crate::result::outward::<S>(
                    e.error
                        + (projected.into_f64() - value).abs()
                        + 16.0 * f64::EPSILON * value.abs(),
                )?),
                slope,
                scioxide_core::AlgorithmId::from_components(
                    algorithm_component!("Kelvin"),
                    e.method,
                    1,
                ),
                used,
                crate::SpecialFunctionTermination::Converged,
            )
            .map_form(Either::Left));
        }
    }
    crate::complex_support::report_with_status::<S>(
        estimate,
        true,
        algorithm_component!("Kelvin"),
        used,
        exhausted,
    )
    .map(|e| e.map_form(Either::Right))
}

fn origin(index: usize, derivative: bool, work: &mut Work) -> Result<Estimate, SciOxideError> {
    work.try_step()?;
    let (value, second) = match index {
        0 => (1.0, 0.0),
        1 => (0.0, 0.5),
        _ => (-core::f64::consts::FRAC_PI_4, 0.0),
    };
    Ok(Estimate::from(crate::support::JetData {
        y: Jet {
            v: real(if derivative { 0.0 } else { value }),
            d: real(if derivative { second } else { 0.0 }),
        },
        scale: Jet::constant(real(0.0)),
        error: 0.0,
        derivative_error: Some(if index == 3 && derivative {
            f64::INFINITY
        } else {
            0.0
        }),
        method: algorithm_component!("Kelvin_origin"),
    }))
}

/// Real order-zero series (DLMF 10.65.1/10.65.2), differentiated termwise.
/// The |x|<=10 bound limits cancellation; larger arguments use Bessel scaling.
/// Independent derivative seeds avoid dividing an underflowed value term by x.
fn growing_series(a: f64, work: &mut Work) -> Result<[Estimate; 2], SciOxideError> {
    use crate::support::wide::Wide;
    work.try_step()?;
    let q = a * a * 0.25;
    let square = -q * q;
    let mut terms = [1.0, q, -a * q * 0.25, a * 0.5];
    let mut sums = terms.map(Wide::from);
    let mut magnitudes = terms.map(f64::abs);
    for k in 1usize.. {
        work.try_step()?;
        let n = 2.0 * k as f64;
        let factors = [
            square / (n * n * (n - 1.0) * (n - 1.0)),
            square / (n * n * (n + 1.0) * (n + 1.0)),
            square / (n * (n + 1.0) * (n + 1.0) * (n + 2.0)),
            square / ((n - 1.0) * n * n * (n + 1.0)),
        ];
        for i in 0..4 {
            terms[i] *= factors[i];
            sums[i] = sums[i].add(Wide::from(terms[i]));
            magnitudes[i] += terms[i].abs();
        }
        let values = sums.map(Wide::value);
        if values.iter().any(|v| !v.is_finite()) {
            return Err(crate::support::instability(
                "Kelvin series became non-finite",
            ));
        }
        // Once ratios decrease below one, the remaining alternating tail is
        // bounded by the last included term. Include recurrence and sum rounding.
        if (0..4)
            .all(|i| factors[i].abs() < 1.0 && terms[i].abs() <= f64::EPSILON * values[i].abs())
        {
            let errors = core::array::from_fn::<_, 4, _>(|i| {
                2.0 * terms[i].abs() + 16.0 * f64::EPSILON * (n + 1.0) * magnitudes[i]
            });
            return Ok(core::array::from_fn(|i| {
                Estimate::from(crate::support::JetData {
                    y: Jet {
                        v: real(values[i]),
                        d: real(values[i + 2]),
                    },
                    scale: Jet::constant(real(0.0)),
                    error: errors[i],
                    derivative_error: Some(errors[i + 2]),
                    method: algorithm_component!("Kelvin_real_series"),
                })
            }));
        }
    }
    Err(crate::support::instability(
        "Kelvin series did not converge",
    ))
}

fn pair_estimates(
    a: f64,
    k: bool,
    scaled: bool,
    work: &mut Work,
) -> Result<[Estimate; 2], SciOxideError> {
    use crate::bessel_airy::complex_bessel_common::Kind;
    if !k && a <= 10.0 {
        let mut pair = growing_series(a, work)?;
        if scaled {
            let multiplier = (-a * core::f64::consts::FRAC_1_SQRT_2).exp();
            for e in &mut pair {
                e.y.v *= multiplier;
                e.y.d *= multiplier;
                e.error = e.error * multiplier + 4.0 * f64::EPSILON * e.y.v.norm();
                e.derivative_error = e
                    .derivative_error
                    .map(|error| error * multiplier + 4.0 * f64::EPSILON * e.y.d.norm());
            }
        }
        return Ok(pair);
    }
    if a >= 1e-150 {
        return super::kelvin_kernel::pair(a, k, scaled, work);
    }
    let ray = C::new(
        core::f64::consts::FRAC_1_SQRT_2,
        core::f64::consts::FRAC_1_SQRT_2,
    );
    let z = ray.scale(a);
    let normalization = if k { -z.re } else { z.re };
    let e = crate::bessel_airy::complex_order_bessel::kernel_value(
        if k { Kind::K } else { Kind::I },
        real(0.0),
        Jet::variable(z),
        real(-normalization),
        work,
    )?;
    let factor = e.scale.v.exp();
    let value = e.y.v * factor;
    let first = ray * (e.y.d + e.scale.d * e.y.v) * factor;
    // Rounding the ray point perturbs its exponential phase by about 2 epsilon a.
    let argument = 2.0 * f64::EPSILON * a;
    let error = e.error * factor.norm() + argument * value.norm();
    let derivative_error =
        e.derivative_error.unwrap_or(e.error) * factor.norm() + argument * first.norm();
    let multiplier = if scaled { 1.0 } else { normalization.exp() };
    let expand = multiplier.is_finite() && multiplier > 0.0;
    Ok(core::array::from_fn(|i| {
        let component = |v: C| if i == 0 { v.re } else { v.im };
        let multiplier = if expand { multiplier } else { 1.0 };
        Estimate::from(crate::support::JetData {
            y: Jet {
                v: real(component(value) * multiplier),
                d: real(component(first) * multiplier),
            },
            scale: Jet::constant(real(if expand { 0.0 } else { normalization })),
            error: error * multiplier,
            derivative_error: Some(derivative_error * multiplier),
            method: algorithm_component!("Kelvin_shared_Bessel"),
        })
    }))
}

fn select(pair: [Estimate; 2], index: usize, derivative: bool, negative: bool, a: f64) -> Estimate {
    let mut e = pair[index];
    if derivative {
        // Both ray solutions obey y'' + y'/x - i y = 0. Scaling multiplies
        // the original derivatives; it is never differentiated here.
        let other = pair[1 - index];
        let second = (if index == 0 {
            -other.y.v.re
        } else {
            other.y.v.re
        }) - e.y.d.re / a;
        let error = e.derivative_error.unwrap_or(e.error);
        e.derivative_error = Some(other.error + error / a + 8.0 * f64::EPSILON * second.abs());
        e.error = error;
        e.y = Jet {
            v: e.y.d,
            d: real(second),
        };
        if negative {
            e.y.v = -e.y.v;
        }
    } else if negative {
        e.y.d = -e.y.d;
    }
    e
}

/// Evaluation of one Kelvin component.
type KelvinEvaluation<S> = Evaluation<S, Either<Form<S>, ScaledForm>>;
/// Build the detailed result of every combined component.
fn publish_all<S: RealScalar>(
    [ber, bei, ker, kei]: [KelvinEvaluation<S>; 4],
) -> Result<[Detailed<S>; 4], SciOxideError> {
    Ok([
        ber.publish()?,
        bei.publish()?,
        ker.publish()?,
        kei.publish()?,
    ])
}
/// Evaluate before projecting the value or building the detailed result.
fn kelvin_evaluate<S: RealScalar>(
    x: S,
    options: KelvinOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<[KelvinEvaluation<S>; 4], SciOxideError> {
    crate::support::require_finite_param(x, "Bessel parameter x must be finite")?;
    if x <= S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "combined Kelvin evaluation requires x > 0",
        ));
    }
    let mut output = [None; 4];
    let a = x.try_to_f64_exact()?;
    let mut work = Work::with_batched_context(options.iteration_budget().get(), context)?;
    for (index, slots) in output.chunks_exact_mut(2).enumerate() {
        if work.remaining == 0 {
            return Err(work.limit_error());
        }
        let before = work.used;
        let pair = pair_estimates(a, index == 1, false, &mut work);
        work.check()?;
        let used = work.used - before;
        if work.exhausted() {
            return Err(work.limit_error());
        }
        let [first, second] = pair?;
        slots[0] = Some(report(Ok(first), used, false)?);
        slots[1] = Some(report(Ok(second), 0, false)?);
    }
    let [ber, bei, ker, kei] = output;
    Ok([
        ber.ok_or_else(|| work.limit_error())?,
        bei.ok_or_else(|| work.limit_error())?,
        ker.ok_or_else(|| work.limit_error())?,
        kei.ok_or_else(|| work.limit_error())?,
    ])
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the four Kelvin functions `[ber(x), bei(x), ker(x), kei(x)]` for `x > 0` in one call.

The definitions are those of [`ber`](crate::ber), [`bei`](crate::bei), [`ker`](crate::ker) and
[`kei`](crate::kei). One evaluation on the ray `x e^(iπ/4)` gives both `ber` and `bei`, and one
gives both `ker` and `kei`, so the call costs about as much as `ber` and `ker` together.
[`kelvin_with_options`](crate::kelvin_with_options) also returns the derivative of each
component.

# Examples

```
// bei(1) = 0.2495660400366597…, ker(1) = 0.2867062087283160…
let [_ber, bei, ker, _kei] = scioxide_special::kelvin(1.0_f64)?;
assert!((bei - 0.249_566_040_036_659_72).abs() < 1e-12);
assert!((ker - 0.286_706_208_728_316_04).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated for Kelvin functions. Tests compare values, derivatives and
error estimates with reference values in `f32` and `f64`.
[`kelvin_with_options`](crate::kelvin_with_options) returns an error estimate for each
argument. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`, where `ker` is singular.
- `DomainError` (`OutsideDomain`) if `x < 0`.
- `NumericalInstability` (`Overflow`) if `|ber(x)|` or `|bei(x)|` exceeds the range of the
  scalar type.

# References

NIST DLMF, §10.61–10.67. <https://dlmf.nist.gov/10.61>"###)]

    pub kelvin [S: RealScalar] (x: S) -> [S; 4]
    where [];
    options options: KelvinOptions;
    detailed kelvin_with_options -> [Detailed<S>; 4];
    value {
        let [ber, bei, ker, kei] = kelvin_evaluate(x, KelvinOptions::default(), None)?;
        Ok([ber.value()?, bei.value()?, ker.value()?, kei.value()?])
    }
    report {
        publish_all(kelvin_evaluate(x, options, None)?)
    }
    scoped pub kelvin_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            kelvin_evaluate(x, options, Some(context))
        })
        .and_then(publish_all)
    }
}
