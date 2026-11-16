//! Bessel/Struve integrals from integrated defining series and Hankel tails.
//! Mellin finite parts fix the logarithmic integration constants (DLMF 10.22).
use crate::expansions::{bounded, real};
use crate::result::Evaluation;
use crate::support::Work;
use crate::{BesselIntegralOptions, SpecialFunctionResult};
use scioxide_core::{AlgorithmComponent, Complex, RealScalar, SciOxideError, algorithm_component};
type Detailed<S> = SpecialFunctionResult<S>;
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    JInt,
    YInt,
    IInt,
    KInt,
    JComplement,
    YTail,
    IRemainder,
    KTail,
    HInt,
    HTail,
    LInt,
}
struct Pair<S> {
    value: S,
    derivative: S,
    error: S,
    derivative_error: S,
}
fn empty<S: RealScalar>() -> Pair<S> {
    Pair {
        value: S::ZERO,
        derivative: S::ZERO,
        error: S::MAX,
        derivative_error: S::MAX,
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the integral `∫₀^x J₀(t) dt` of the Bessel function of the first kind.

The integral is odd in `x` and defined for every finite `x`. It tends to 1 as `x → ∞`.

For `|x| ≤ 12` the function sums the termwise-integrated power series. On `12 < |x| < 32` it
evaluates Chebyshev expansions, and from `|x| = 32` it uses the asymptotic expansion of the
tail integral.

# Examples

```
// ∫₀^x J₀(t) dt is odd in x and vanishes at x = 0.
let positive = scioxide_special::bessel_j0_integral(1.0_f64)?;
assert!((scioxide_special::bessel_j0_integral(-1.0_f64)? + positive).abs() < 1e-13);
assert!(scioxide_special::bessel_j0_integral(0.0_f64)?.abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`bessel_j0_integral_with_options`](crate::bessel_j0_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"Every finite `x` is accepted; only the general conditions above apply.

# References

NIST DLMF, §10.22 and §10.43. <https://dlmf.nist.gov/10.22>

M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 11."###)]

    pub bessel_j0_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed bessel_j0_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(bessel_j0_integral));
        evaluate(x, options, Kind::JInt, family, context)
    }
    scoped pub bessel_j0_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the integral `∫₀^x Y₀(t) dt` of the Bessel function of the second kind for `x ≥ 0`.

The logarithmic singularity of `Y₀` at the origin is integrable, so the integral is zero at
`x = 0`. It tends to 0 as `x → ∞`.

For `|x| ≤ 12` the function sums the termwise-integrated power series. On `12 < |x| < 32` it
evaluates Chebyshev expansions, and from `|x| = 32` it uses the asymptotic expansion of the
tail integral.

# Examples

```
// ∫₀¹ Y₀(t) dt = −0.6370693766074231…
let i = scioxide_special::bessel_y0_integral(1.0_f64)?;
assert!((i + 0.637_069_376_607_423_1).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`bessel_y0_integral_with_options`](crate::bessel_y0_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.22 and §10.43. <https://dlmf.nist.gov/10.22>

M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 11."###)]

    pub bessel_y0_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed bessel_y0_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(bessel_y0_integral));
        evaluate(x, options, Kind::YInt, family, context)
    }
    scoped pub bessel_y0_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the integral `∫₀^x I₀(t) dt` of the modified Bessel function of the first kind.

The integral is odd in `x` and defined for every finite `x`. It grows like
`exp(|x|)/√(2π|x|)`.

For `|x| < 32` the function sums the termwise-integrated power series; from `|x| = 32` it
uses the asymptotic expansion of the integral.

# Examples

```
// ∫₀¹ I₀(t) dt = 1.08652109702359…
let i = scioxide_special::bessel_i0_integral(1.0_f64)?;
assert!((i - 1.086_521_097_023_59).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`bessel_i0_integral_with_options`](crate::bessel_i0_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|∫₀^x I₀(t) dt|` exceeds the range of the scalar type.

# References

NIST DLMF, §10.22 and §10.43. <https://dlmf.nist.gov/10.22>

M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 11."###)]

    pub bessel_i0_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed bessel_i0_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(bessel_i0_integral));
        evaluate(x, options, Kind::IInt, family, context)
    }
    scoped pub bessel_i0_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the integral `∫₀^x K₀(t) dt` of the modified Bessel function of the second kind for `x ≥ 0`.

The logarithmic singularity of `K₀` at the origin is integrable, so the integral is zero at
`x = 0`. It tends to `π/2` as `x → ∞`.

For `x ≤ 12` the function sums the termwise-integrated power series. On `12 < x < 32` it
evaluates Chebyshev expansions, and from `x = 32` it subtracts the asymptotic expansion of
`∫ₓ^∞ K₀(t) dt` from `π/2`.

# Examples

```
// ∫₀¹ K₀(t) dt = 1.2425098486237782…
let i = scioxide_special::bessel_k0_integral(1.0_f64)?;
assert!((i - 1.242_509_848_623_778_2).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`bessel_k0_integral_with_options`](crate::bessel_k0_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.22 and §10.43. <https://dlmf.nist.gov/10.22>

M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 11."###)]

    pub bessel_k0_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed bessel_k0_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(bessel_k0_integral));
        evaluate(x, options, Kind::KInt, family, context)
    }
    scoped pub bessel_k0_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the integral `∫₀^x (1 − J₀(t))/t dt`.

The integrand is regular at the origin and even, so the integral is even in `x` and zero at
`x = 0`. Every finite `x` is accepted. For large `x` it grows like `ln(x/2) + γ`,
where `γ` is Euler's constant.

For `|x| ≤ 12` the function sums the termwise-integrated power series. On `12 < |x| < 32` it
evaluates Chebyshev expansions, and from `|x| = 32` it uses the asymptotic expansion of the
tail integral.

# Examples

```
// ∫₀¹ (1 − J₀(t))/t dt = 0.12116524699506871…
let i = scioxide_special::bessel_j0_complement_integral(1.0_f64)?;
assert!((i - 0.121_165_246_995_068_71).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`bessel_j0_complement_integral_with_options`](crate::bessel_j0_complement_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"Every finite `x` is accepted; only the general conditions above apply.

# References

NIST DLMF, §10.22 and §10.43. <https://dlmf.nist.gov/10.22>

M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 11."###)]

    pub bessel_j0_complement_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed bessel_j0_complement_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(bessel_j0_complement_integral));
        evaluate(x, options, Kind::JComplement, family, context)
    }
    scoped pub bessel_j0_complement_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the tail integral `∫ₓ^∞ Y₀(t)/t dt` for `x > 0`.

The improper integral converges at infinity. It diverges as `x → 0⁺`.

For `|x| ≤ 12` the function sums the termwise-integrated power series. On `12 < |x| < 32` it
evaluates Chebyshev expansions, and from `|x| = 32` it uses the asymptotic expansion of the
tail integral.

# Examples

```
// ∫₁^∞ Y₀(t)/t dt = 0.3952729016992933…
let i = scioxide_special::bessel_y0_tail_integral(1.0_f64)?;
assert!((i - 0.395_272_901_699_293_3).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`bessel_y0_tail_integral_with_options`](crate::bessel_y0_tail_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`, where the integral diverges.
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.22 and §10.43. <https://dlmf.nist.gov/10.22>

M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 11."###)]

    pub bessel_y0_tail_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed bessel_y0_tail_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(bessel_y0_tail_integral));
        evaluate(x, options, Kind::YTail, family, context)
    }
    scoped pub bessel_y0_tail_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the integral `∫₀^x (I₀(t) − 1)/t dt`.

The integrand is regular at the origin and even, so the integral is even in `x` and zero at
`x = 0`. Every finite `x` is accepted. It grows like `exp(|x|)/(|x|√(2π|x|))`.

For `|x| < 32` the function sums the termwise-integrated power series; from `|x| = 32` it
uses the asymptotic expansion of the integral.

# Examples

```
// ∫₀¹ (I₀(t) − 1)/t dt = 0.12897944249456852…
let i = scioxide_special::bessel_i0_remainder_integral(1.0_f64)?;
assert!((i - 0.128_979_442_494_568_52).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`bessel_i0_remainder_integral_with_options`](crate::bessel_i0_remainder_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|∫₀^x (I₀(t) − 1)/t dt|` exceeds the range of the scalar type.

# References

NIST DLMF, §10.22 and §10.43. <https://dlmf.nist.gov/10.22>

M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 11."###)]

    pub bessel_i0_remainder_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed bessel_i0_remainder_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(bessel_i0_remainder_integral));
        evaluate(x, options, Kind::IRemainder, family, context)
    }
    scoped pub bessel_i0_remainder_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the tail integral `∫ₓ^∞ K₀(t)/t dt` for `x > 0`.

The integral decays like `exp(−x)/x^(3/2)` for large `x` and diverges as `x → 0⁺`.

For `x ≤ 12` the function uses the termwise-integrated power series. On `12 < x < 32` it
evaluates Chebyshev expansions, and from `x = 32` it sums the asymptotic expansion.

# Examples

```
// ∫₁^∞ K₀(t)/t dt = 0.2085182909001295…
let i = scioxide_special::bessel_k0_tail_integral(1.0_f64)?;
assert!((i - 0.208_518_290_900_129_5).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`bessel_k0_tail_integral_with_options`](crate::bessel_k0_tail_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`, where the integral diverges.
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

NIST DLMF, §10.22 and §10.43. <https://dlmf.nist.gov/10.22>

M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 11."###)]

    pub bessel_k0_tail_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed bessel_k0_tail_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(bessel_k0_tail_integral));
        evaluate(x, options, Kind::KTail, family, context)
    }
    scoped pub bessel_k0_tail_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the integral `∫₀^x H₀(t) dt` of the Struve function [`struve`](crate::struve) of order zero.

`H₀` is odd, so the integral is even in `x` and zero at `x = 0`. Every finite `x`
is accepted.

For `|x| ≤ 12` the function sums the termwise-integrated power series. On `12 < |x| < 32` it
evaluates Chebyshev expansions, and from `|x| = 32` it uses the asymptotic expansion of the
tail integral.

# Examples

```
// ∫₀¹ H₀(t) dt = 0.30109042670805547…
let i = scioxide_special::struve_h0_integral(1.0_f64)?;
assert!((i - 0.301_090_426_708_055_47).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`struve_h0_integral_with_options`](crate::struve_h0_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"Every finite `x` is accepted; only the general conditions above apply.

# References

NIST DLMF, §11.7. <https://dlmf.nist.gov/11.7>"###)]

    pub struve_h0_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed struve_h0_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(struve_h0_integral));
        evaluate(x, options, Kind::HInt, family, context)
    }
    scoped pub struve_h0_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the tail integral `∫ₓ^∞ H₀(t)/t dt` of the Struve function of order zero.

The improper integral converges for every real `x`. Its value at `x = 0` is `π/2`, and since
`H₀(t)/t` is even, `∫₋ₓ^∞ H₀(t)/t dt = π − ∫ₓ^∞ H₀(t)/t dt`.

For `|x| ≤ 12` the function sums the termwise-integrated power series. On `12 < |x| < 32` it
evaluates Chebyshev expansions, and from `|x| = 32` it uses the asymptotic expansion of the
tail integral.

# Examples

```
// ∫₀^∞ H₀(t)/t dt = π/2
let i = scioxide_special::struve_h0_tail_integral(0.0_f64)?;
assert!((i - core::f64::consts::FRAC_PI_2).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`struve_h0_tail_integral_with_options`](crate::struve_h0_tail_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"Every finite `x` is accepted; only the general conditions above apply.

# References

NIST DLMF, §11.7. <https://dlmf.nist.gov/11.7>"###)]

    pub struve_h0_tail_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed struve_h0_tail_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(struve_h0_tail_integral));
        evaluate(x, options, Kind::HTail, family, context)
    }
    scoped pub struve_h0_tail_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the integral `∫₀^x L₀(t) dt` of the modified Struve function [`modstruve`](crate::modstruve) of order zero.

`L₀` is odd, so the integral is even in `x` and zero at `x = 0`. Every finite `x`
is accepted. It grows like `exp(|x|)/√(2π|x|)`.

For `|x| < 32` the function sums the termwise-integrated power series; from `|x| = 32` it
uses the asymptotic expansion of the integral.

# Examples

```
// ∫₀¹ L₀(t) dt = 0.3364726286440384…
let i = scioxide_special::struve_l0_integral(1.0_f64)?;
assert!((i - 0.336_472_628_644_038_4).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`struve_l0_integral_with_options`](crate::struve_l0_integral_with_options)
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|∫₀^x L₀(t) dt|` exceeds the range of the scalar type.

# References

NIST DLMF, §11.7. <https://dlmf.nist.gov/11.7>"###)]

    pub struve_l0_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: BesselIntegralOptions;
    detailed struve_l0_integral_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(struve_l0_integral));
        evaluate(x, options, Kind::LInt, family, context)
    }
    scoped pub struve_l0_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

pub(crate) fn evaluate<S: RealScalar>(
    x: S,
    options: BesselIntegralOptions,
    kind: Kind,
    family: AlgorithmComponent,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    evaluate_charged(x, kind, family, &mut work)
}

/// Evaluate one integral charged to the caller's allowance, which may be shared.
pub(crate) fn evaluate_charged<S: RealScalar>(
    x: S,
    kind: Kind,
    family: AlgorithmComponent,
    work: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "argument must be finite",
        ));
    }
    if matches!(kind, Kind::YInt | Kind::KInt | Kind::YTail | Kind::KTail) && x < S::ZERO
        || matches!(kind, Kind::YTail | Kind::KTail) && x == S::ZERO
    {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "outside the real integral domain",
        ));
    }
    let p = kernel(kind, x.abs().try_to_f64_exact()?, work);
    work.check()?;
    let mut p = p?;
    if x.is_sign_negative() {
        match kind {
            Kind::JInt | Kind::IInt => p.value = -p.value,
            Kind::JComplement | Kind::IRemainder | Kind::HInt | Kind::LInt => {
                p.derivative = -p.derivative
            }
            Kind::HTail => p.value = core::f64::consts::PI - p.value,
            _ => {}
        }
    }
    crate::support::check_finite(p.value)?;
    let maximum = S::MAX.try_to_f64_exact()?;
    let value = crate::support::project_real::<S>(p.value)?;
    let error =
        (S::try_from_f64_lossy(p.error.min(maximum))? + S::EPSILON * value.abs()).min(S::MAX);
    let derivative = if p.derivative.is_finite() && S::try_from_f64_lossy(p.derivative).is_ok() {
        let d = S::try_from_f64_lossy(p.derivative)?;
        Some((
            d,
            crate::result::Bound::heuristic(
                (S::try_from_f64_lossy(p.derivative_error.min(maximum))? + S::EPSILON * d.abs())
                    .min(S::MAX),
            ),
        ))
    } else {
        None
    };
    Ok(bounded(
        value,
        crate::result::Bound::derived(error),
        scioxide_core::AlgorithmId::from_components(
            family,
            algorithm_component!("integrated_series_tail"),
            1,
        ),
        work.used,
        work.complete,
    )?
    .with_derivative(derivative))
}
fn kernel<S: RealScalar>(kind: Kind, x: S, w: &mut Work) -> Result<Pair<S>, SciOxideError> {
    if x == S::ZERO {
        let value = if matches!(kind, Kind::HTail) {
            S::PI * S::HALF
        } else {
            S::ZERO
        };
        let derivative = match kind {
            Kind::JInt | Kind::IInt => S::ONE,
            Kind::YInt => S::NEG_INFINITY,
            Kind::KInt => S::INFINITY,
            Kind::HTail => -real::<S>(2.0)? / S::PI,
            _ => S::ZERO,
        };
        return Ok(Pair {
            value,
            derivative,
            error: S::EPSILON * value.abs(),
            derivative_error: S::EPSILON * derivative.abs(),
        });
    }
    if matches!(kind, Kind::KInt | Kind::KTail) && x >= real::<S>(32.0)? {
        return k_tail(kind, x, w);
    }
    if x <= real::<S>(12.0)?
        || matches!(kind, Kind::IInt | Kind::IRemainder | Kind::LInt) && x < real::<S>(32.0)?
    {
        return power(kind, x, w);
    }
    let table = match kind {
        Kind::JInt => Some(0),
        Kind::YInt => Some(1),
        Kind::JComplement => Some(2),
        Kind::YTail => Some(3),
        Kind::HInt => Some(4),
        Kind::HTail => Some(5),
        Kind::KInt => Some(6),
        Kind::KTail => Some(7),
        _ => None,
    };
    if x < real::<S>(32.0)?
        && let Some(table) = table
    {
        return interpolation(table, x, w);
    }
    if matches!(kind, Kind::IInt | Kind::IRemainder | Kind::LInt) {
        return i_tail(kind, x, w);
    }
    let weighted = matches!(kind, Kind::JComplement | Kind::YTail | Kind::HTail);
    let (tail, error) = oscillatory_tail(x, usize::from(weighted), w)?;
    let j = crate::bessel_airy::bessel::bessel_j(S::ZERO, x, w)?;
    let y = crate::bessel_airy::bessel::bessel_y(S::ZERO, x, w)?;
    let two_pi = real::<S>(2.0)? / S::PI;
    let logarithm = x.ln() + real::<S>(2.0)?.ln() + real::<S>(0.577_215_664_901_532_9)?;
    let (value, derivative, extra) = match kind {
        Kind::JInt => (S::ONE - tail.re, j, S::EPSILON),
        Kind::YInt => (-tail.im, y, S::ZERO),
        Kind::JComplement => (
            logarithm - real::<S>(2.0)? * real::<S>(2.0)?.ln() + tail.re,
            (S::ONE - j) / x,
            S::EPSILON * logarithm.abs(),
        ),
        Kind::YTail => (tail.im, -y / x, S::ZERO),
        Kind::HInt | Kind::HTail => {
            let (sum, integral, last) = struve_tail(x, false, weighted, w)?;
            let h = y + two_pi / x * sum;
            if weighted {
                (tail.im + two_pi / x * integral, -h / x, two_pi / x * last)
            } else {
                (two_pi * (logarithm - integral) - tail.im, h, two_pi * last)
            }
        }
        _ => return Ok(empty()),
    };
    Ok(Pair {
        value,
        derivative,
        error: error + extra + real::<S>(32.0)? * S::EPSILON * value.abs(),
        // The derivative combines the oracle-qualified 64-unit J0/Y0 envelope
        // with the value's truncation, scaled by the 1/x weights.
        derivative_error: real::<S>(64.0)?
            * S::EPSILON
            * (j.abs() + y.abs() + S::ONE)
            * (S::ONE + S::ONE / x)
            + extra / x
            + real::<S>(32.0)? * S::EPSILON * derivative.abs(),
    })
}
fn power<S: RealScalar>(kind: Kind, x: S, w: &mut Work) -> Result<Pair<S>, SciOxideError> {
    let l = x.ln() - real::<S>(2.0)?.ln() + real::<S>(0.577_215_664_901_532_9)?;
    let two_pi = real::<S>(2.0)? / S::PI;
    let struve = matches!(kind, Kind::HInt | Kind::HTail | Kind::LInt);
    let positive = matches!(
        kind,
        Kind::IInt | Kind::KInt | Kind::IRemainder | Kind::KTail | Kind::LInt
    );
    let mut value = match kind {
        Kind::JInt | Kind::IInt => x,
        Kind::YInt => two_pi * x * (l - S::ONE),
        Kind::KInt => x * (S::ONE - l),
        Kind::YTail => S::PI / real(6.0)? - l * l / S::PI,
        Kind::KTail => S::PI * S::PI / real(24.0)? + l * l * S::HALF,
        Kind::HInt | Kind::LInt => x * x / S::PI,
        Kind::HTail => S::PI * S::HALF - two_pi * x,
        _ => S::ZERO,
    };
    let mut derivative = match kind {
        Kind::JInt | Kind::IInt => S::ONE,
        Kind::YInt => two_pi * l,
        Kind::KInt => -l,
        Kind::YTail => -two_pi * l / x,
        Kind::KTail => l / x,
        Kind::HInt | Kind::LInt => two_pi * x,
        Kind::HTail => -two_pi,
        _ => S::ZERO,
    };
    let mut term = S::ONE;
    let mut harmonic = S::ZERO;
    let mut absolute = value.abs();
    let mut d_absolute = derivative.abs();
    let mut last = S::ONE;
    let mut d_last = S::ONE;
    let mut k = 0usize;
    while w.step() {
        k += 1;
        let ks = S::try_from_usize_exact(k)?;
        let even = real::<S>(2.0)? * ks;
        let odd = even + S::ONE;
        term =
            term * if struve {
                x * x / (odd * odd)
            } else {
                (x * S::HALF / ks) * (x * S::HALF / ks)
            } * if positive { S::ONE } else { -S::ONE };
        harmonic += S::ONE / ks;
        let (v, d) = match kind {
            Kind::JInt | Kind::IInt => (x * term / odd, term),
            Kind::YInt => (
                two_pi * x * term / odd * (l - harmonic - S::ONE / odd),
                two_pi * term * (l - harmonic),
            ),
            Kind::KInt => (
                x * term / odd * (-l + harmonic + S::ONE / odd),
                term * (-l + harmonic),
            ),
            Kind::JComplement => (
                -term / even,
                if k == 1 { x / real(4.0)? } else { -term / x },
            ),
            Kind::IRemainder => (term / even, if k == 1 { x / real(4.0)? } else { term / x }),
            Kind::YTail => (
                -two_pi * term / even * (l - harmonic - S::ONE / even),
                -two_pi * term * (l - harmonic) / x,
            ),
            Kind::KTail => (
                term / even * (l - harmonic - S::ONE / even),
                term * (l - harmonic) / x,
            ),
            Kind::HInt | Kind::LInt => (x * x / S::PI * term / (ks + S::ONE), two_pi * x * term),
            Kind::HTail => (-two_pi * x * term / odd, -two_pi * term),
        };
        value += v;
        derivative += d;
        absolute += v.abs();
        d_absolute += d.abs();
        last = v.abs();
        d_last = d.abs();
        if k >= 4 && last <= S::EPSILON * value.abs() && d_last <= S::EPSILON * derivative.abs() {
            break;
        }
    }
    Ok(Pair {
        value,
        derivative,
        error: last + real::<S>(32.0)? * S::EPSILON * absolute,
        derivative_error: d_last + real::<S>(32.0)? * S::EPSILON * d_absolute,
    })
}
fn interpolation<S: RealScalar>(
    table: usize,
    x: S,
    w: &mut Work,
) -> Result<Pair<S>, SciOxideError> {
    use crate::bessel_airy::bessel_integral_coefficients::{EDGES, TABLES};
    let input = x.try_to_f64_exact()?;
    let segment = EDGES.windows(2).position(|p| input <= p[1]).unwrap_or(2);
    let c = &TABLES[table][segment];
    if w.remaining < 2 * c.len() {
        w.complete = false;
        return Ok(empty());
    }
    let lo = real::<S>(EDGES[segment])?;
    let hi = real::<S>(EDGES[segment + 1])?;
    let t = (real::<S>(2.0)? * x - lo - hi) / (hi - lo);
    let (value, error) = crate::expansions::chebyshev(t, c, w)?;
    let mut bound = S::ZERO;
    let (first, _) = crate::support::polynomial::clenshaw(
        c.iter().enumerate().skip(1),
        S::ZERO,
        |first, second, (index, &coefficient)| {
            w.try_step()?;
            let k = S::try_from_usize_exact(index)?;
            let coefficient = real::<S>(coefficient)?;
            bound += k * k * coefficient.abs();
            Ok::<_, SciOxideError>(S::TWO * t * first - second + k * coefficient)
        },
    )?;
    let derivative = real::<S>(2.0)? * first / (hi - lo);
    Ok(Pair {
        value,
        derivative,
        error,
        derivative_error: real::<S>(32.0)? * S::EPSILON * bound / (hi - lo),
    })
}
fn coefficients<S: RealScalar>(
    x: S,
    power: usize,
    rotation: Complex<S>,
    w: &mut Work,
) -> Result<(Complex<S>, S), SciOxideError> {
    let mut a = S::ONE;
    let mut c = S::ONE;
    let mut phase = Complex::new(S::ONE, S::ZERO);
    let mut sum = phase;
    let mut previous = S::MAX;
    let mut omitted = S::ONE;
    let mut k = 0usize;
    while w.step() {
        k += 1;
        let ks = S::try_from_usize_exact(k)?;
        let odd = real::<S>(2.0)? * ks - S::ONE;
        a = a * (odd / x) * (odd / (real::<S>(8.0)? * ks));
        c = ((S::try_from_usize_exact(power)? + ks - S::HALF) / x) * c + a;
        omitted = c.abs();
        if omitted > previous {
            break;
        }
        phase *= rotation;
        sum += phase.scale(c);
        previous = omitted;
        if omitted <= S::EPSILON * sum.norm() {
            break;
        }
    }
    Ok((sum, omitted + real::<S>(32.0)? * S::EPSILON * sum.norm()))
}
fn oscillatory_tail<S: RealScalar>(
    x: S,
    power: usize,
    w: &mut Work,
) -> Result<(Complex<S>, S), SciOxideError> {
    let (sum, error) = coefficients(x, power, Complex::new(S::ZERO, -S::ONE), w)?;
    let (s, c) = x.sin_cos();
    let root = real::<S>(2.0)?.sqrt();
    let phase = Complex::new((c + s) / root, (s - c) / root);
    let mut prefactor = (real::<S>(2.0)? / S::PI).sqrt() / x.sqrt();
    if power == 1 {
        prefactor /= x;
    }
    Ok((
        phase * Complex::new(S::ZERO, S::ONE) * sum.scale(prefactor),
        error * prefactor,
    ))
}
fn k_tail<S: RealScalar>(kind: Kind, x: S, w: &mut Work) -> Result<Pair<S>, SciOxideError> {
    let weighted = matches!(kind, Kind::KTail);
    let (sum, error) = coefficients(x, usize::from(weighted), Complex::new(-S::ONE, S::ZERO), w)?;
    let mut prefactor = (S::PI * S::HALF).sqrt() / x.sqrt() * (-x).exp();
    if weighted {
        prefactor /= x;
    }
    let tail = prefactor * sum.re;
    let k0 = crate::bessel_airy::bessel::bessel_k_scaled(S::ZERO, x, w)? * (-x).exp();
    let value = if weighted {
        tail
    } else {
        S::PI * S::HALF - tail
    };
    let derivative = if weighted { -k0 / x } else { k0 };
    Ok(Pair {
        value,
        derivative,
        error: prefactor * error + real::<S>(32.0)? * S::EPSILON * value.abs(),
        // The 64-unit K0 envelope and the rounding of exp(-x) and 1/x.
        derivative_error: real::<S>(96.0)? * S::EPSILON * derivative.abs(),
    })
}
fn exp_product<S: RealScalar>(value: S, x: S) -> S {
    let factor = (x * S::HALF).exp();
    (value * factor) * factor
}
fn i_tail<S: RealScalar>(kind: Kind, x: S, w: &mut Work) -> Result<Pair<S>, SciOxideError> {
    let weighted = matches!(kind, Kind::IRemainder);
    let (sum, error) = coefficients(x, usize::from(weighted), Complex::new(S::ONE, S::ZERO), w)?;
    let mut prefactor = S::ONE / ((real::<S>(2.0)? * S::PI).sqrt() * x.sqrt());
    if weighted {
        prefactor /= x;
    }
    let mut value = exp_product(prefactor * sum.re, x);
    let i0 = crate::bessel_airy::bessel::bessel_i_scaled(S::ZERO, x, w)?;
    let growing = exp_product(if weighted { i0 / x } else { i0 }, x);
    let mut derivative = growing - if weighted { S::ONE / x } else { S::ZERO };
    let mut derivative_error = real::<S>(96.0)?
        * S::EPSILON
        * (growing.abs() + if weighted { S::ONE / x } else { S::ZERO });
    let l = x.ln() - real::<S>(2.0)?.ln() + real::<S>(0.577_215_664_901_532_9)?;
    let mut uncertainty = exp_product(prefactor * error, x) + real::<S>(2.0)? * (S::ONE + l.abs());
    if weighted {
        value -= l;
    }
    if matches!(kind, Kind::LInt) {
        let (sum, integral, last) = struve_tail(x, true, false, w)?;
        value -= real::<S>(2.0)? / S::PI * (l + real::<S>(2.0)? * real::<S>(2.0)?.ln() - integral);
        let struve = real::<S>(2.0)? / (S::PI * x) * sum;
        derivative -= struve;
        derivative_error = derivative_error
            + real::<S>(32.0)? * S::EPSILON * struve.abs()
            + real::<S>(2.0)? / (S::PI * x) * last;
        uncertainty += last;
    }
    Ok(Pair {
        value,
        derivative,
        error: uncertainty + real::<S>(32.0)? * S::EPSILON * value.abs(),
        derivative_error,
    })
}
fn struve_tail<S: RealScalar>(
    x: S,
    modified: bool,
    weighted: bool,
    w: &mut Work,
) -> Result<(S, S, S), SciOxideError> {
    let mut term = S::ONE;
    let mut sum = term;
    let mut integral = if weighted { S::ONE } else { S::ZERO };
    let mut old = S::MAX;
    let mut omitted = S::ONE;
    let mut k = 0usize;
    while w.step() {
        k += 1;
        let ks = S::try_from_usize_exact(k)?;
        let odd = real::<S>(2.0)? * ks - S::ONE;
        term = term * (odd / x) * (odd / x) * if modified { S::ONE } else { -S::ONE };
        omitted = term.abs();
        if omitted > old {
            break;
        }
        sum += term;
        integral += term / (real::<S>(2.0)? * ks + if weighted { S::ONE } else { S::ZERO });
        old = omitted;
        if omitted <= S::EPSILON * sum.abs() {
            break;
        }
    }
    Ok((
        sum,
        integral,
        omitted + real::<S>(32.0)? * S::EPSILON * (sum.abs() + integral.abs()),
    ))
}
