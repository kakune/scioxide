//! Coulomb radial waves and hydrogenic states.
//!
//! Positive real waves use the outgoing expansion when its computed value and
//! derivative tails converge and its two-word rounding error stays below one
//! `f64` unit, with no fixed radius threshold; this accepts initial term growth up
//! to about (l²+η²)/x ≈ 59. Otherwise Steed's continued fractions, whose work
//! grows like x − η, or turning-point continuation run under the same budget.
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, ZERO, real};
use crate::support::{is_real, widen_numeric as widen};
mod kernel;
mod real;
mod sequence;

use crate::{CoulombOptions, SpecialFunctionResult};
use scioxide_core::algorithm_component;
use scioxide_core::{Argument, NumericScalar, RealScalar, SciOxideError, SolveContext};
pub use sequence::*;

/// Holds `F_l(η,z)`, `G_l(η,z)` and their radial derivatives from one evaluation.
///
/// Returned by [`crate::coulomb_wave_with_options`] and written by
/// [`crate::coulomb_wave_sequence_into`]. Each component is a [`SpecialFunctionResult`] with its
/// own error estimate and scaled and logarithmic forms. All four components report the
/// counts of the same evaluation, so do not add them together.
///
/// # Examples
///
/// ```
/// // F₀(0, z) = sin z and G₀(0, z) = cos z
/// let waves = scioxide_special::coulomb_wave_with_options(
///     0.0_f64,
///     0.0,
///     1.0_f64,
///     scioxide_special::CoulombOptions::default(),
/// )?;
/// assert!(waves.is_success());
/// assert!((waves.f().try_value()? - 1.0_f64.sin()).abs() < 1e-12);
/// assert!((waves.g().try_value()? - 1.0_f64.cos()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CoulombWaveResult<T: NumericScalar> {
    /// Regular wave F_l(eta,z).
    f: SpecialFunctionResult<T>,
    /// Irregular wave G_l(eta,z).
    g: SpecialFunctionResult<T>,
    /// Radial derivative of F, including scaled/logarithmic forms.
    f_prime: SpecialFunctionResult<T>,
    /// Radial derivative of G, including scaled/logarithmic forms.
    g_prime: SpecialFunctionResult<T>,
}
impl<T: NumericScalar> CoulombWaveResult<T> {
    /// Returns four exact zeros with zero counts.
    ///
    /// Use it to initialize the output and staging buffers of
    /// [`crate::coulomb_wave_sequence_into`]; it is not an evaluation of a Coulomb wave.
    ///
    /// ```
    /// let output = [scioxide_special::CoulombWaveResult::<f64>::zero(); 8];
    /// assert_eq!(output[0].f().counts().iterations(), 0);
    /// ```
    pub fn zero() -> Self {
        let zero = SpecialFunctionResult::zero();
        Self {
            f: zero,
            g: zero,
            f_prime: zero,
            g_prime: zero,
        }
    }

    /// Returns the regular wave `F_l(η,z)`.
    pub const fn f(&self) -> SpecialFunctionResult<T> {
        self.f
    }
    /// Returns the irregular wave `G_l(η,z)`.
    pub const fn g(&self) -> SpecialFunctionResult<T> {
        self.g
    }
    /// Returns the radial derivative `dF_l(η,z)/dz`.
    pub const fn f_prime(&self) -> SpecialFunctionResult<T> {
        self.f_prime
    }
    /// Returns the radial derivative `dG_l(η,z)/dz`.
    pub const fn g_prime(&self) -> SpecialFunctionResult<T> {
        self.g_prime
    }
    /// Returns `true` if all four components terminated successfully.
    pub fn is_success(&self) -> bool {
        [&self.f, &self.g, &self.f_prime, &self.g_prime]
            .iter()
            .all(|r| r.termination().is_success())
    }
}

#[derive(Clone, Copy)]
enum Request {
    Wave,
    Component(Component),
    Normalization,
}

/// Convert once and validate the complete request before opening an execution epoch.
///
/// The orders `l + k` for `k < len` must be finite and distinct; otherwise the error names
/// `length`, the argument that sets the sequence length.
fn parameters<T: NumericScalar>(
    request: Request,
    l: T::Real,
    eta: T::Real,
    z: Option<T>,
    len: usize,
    length: Argument,
) -> Result<(f64, f64, C), SciOxideError> {
    let order = crate::support::widen_numeric(l, "Coulomb parameter l must be finite")?.re;
    let eta = crate::support::widen_numeric(eta, "Coulomb parameter eta must be finite")?.re;
    let point = match z {
        Some(z) => widen(z, "Coulomb parameter z must be finite")?,
        None => ZERO,
    };
    let minimum = if matches!(request, Request::Normalization) {
        -1.0
    } else {
        -0.5
    };
    if order <= minimum {
        return Err(crate::support::COULOMB_ERRORS.outside(scioxide_core::argument!("l")));
    }
    if !matches!(request, Request::Normalization) && is_real::<T>() && point.re < 0.0 {
        return Err(crate::support::COULOMB_ERRORS.outside(scioxide_core::argument!("z")));
    }
    let mut previous = l;
    for k in 0..len.max(1) {
        let next = l + T::Real::try_from_usize_exact(k)?;
        if !next.is_finite() || (k > 0 && next == previous) {
            return Err(crate::support::COULOMB_ERRORS.outside(length));
        }
        previous = next;
        let current = order + k as f64;
        let allowed = match request {
            Request::Normalization => true,
            Request::Wave => current == 0.0 && eta == 0.0,
            Request::Component(Component::F) => true,
            Request::Component(Component::FPrime | Component::SphericalF) => current >= 0.0,
            Request::Component(Component::G) => current == 0.0,
            Request::Component(Component::GPrime) => current == 0.0 && eta == 0.0,
        };
        if point == ZERO && !allowed {
            let pole = current >= 0.0
                && matches!(
                    request,
                    Request::Wave | Request::Component(Component::G | Component::GPrime)
                );
            return Err(crate::support::COULOMB_ERRORS.domain(
                scioxide_core::argument!("z"),
                if pole {
                    scioxide_core::DomainReason::Pole
                } else {
                    scioxide_core::DomainReason::OutsideDomain
                },
            ));
        }
    }
    Ok((order, eta, point))
}

fn validate<T: NumericScalar>(
    l: T::Real,
    eta: T::Real,
    z: T,
) -> Result<(f64, f64, C), SciOxideError> {
    parameters::<T>(
        Request::Wave,
        l,
        eta,
        Some(z),
        1,
        scioxide_core::argument!("l"),
    )
}
/// One kernel evaluation of `[F, G, F′, G′]`, shared by the plain and detailed forms.
fn wave_estimates(
    l: f64,
    eta: f64,
    z: C,
    w: &mut Work<'_, '_, '_>,
) -> Result<[Estimate; 4], SciOxideError> {
    if z == ZERO {
        return Ok([
            origin(Component::F, l, eta, w)?,
            origin(Component::G, l, eta, w)?,
            origin(Component::FPrime, l, eta, w)?,
            origin(Component::GPrime, l, eta, w)?,
        ]);
    }
    let (f, g) = kernel::waves(l, eta, z, w)?;
    let method = algorithm_component!("scaled_coulomb");
    Ok([
        f.estimate(false, method),
        g.estimate(false, method),
        f.estimate(true, method),
        g.estimate(true, method),
    ])
}
fn wave_run<T: NumericScalar>(
    l: T::Real,
    eta: T::Real,
    z: T,
    options: CoulombOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<CoulombWaveResult<T>, SciOxideError> {
    let (l, eta, z) = validate::<T>(l, eta, z)?;
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    w.check()?;
    let estimates = wave_estimates(l, eta, z, &mut w);
    let report = |estimate, derivative| {
        crate::complex_support::report::<T>(
            estimate,
            derivative,
            &w,
            algorithm_component!("coulomb"),
        )?
        .publish()
    };
    let [f, g, f_prime, g_prime] = match estimates {
        Ok([f, g, f_prime, g_prime]) => [
            report(Ok(f), true)?,
            report(Ok(g), true)?,
            report(Ok(f_prime), false)?,
            report(Ok(g_prime), false)?,
        ],
        // Exhaustion shares one interrupted report; other failures propagate.
        Err(e) => [report(Err(e), true)?; 4],
    };
    Ok(CoulombWaveResult {
        f,
        g,
        f_prime,
        g_prime,
    })
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Coulomb wave functions `F_l(η,z)`, `G_l(η,z)` and their derivatives with respect to `z`.

The Coulomb waves solve the radial Coulomb equation (DLMF 33.2.1)

```text
w″ + (1 − 2η/z − l(l+1)/z²) w = 0,
```

where `l` is the order, `η` the charge (Sommerfeld) parameter and `z` the radius. The
regular solution behaves as `F_l ~ C_l(η) z^(l+1)` as `z → 0`, with `C_l(η)` from
[`crate::coulomb_normalization`]. For large real `z`, `F_l ~ sin θ` and `G_l ~ cos θ` with
`θ = z − η ln(2z) − lπ/2 + σ_l(η)` and `σ_l(η) = arg Γ(l+1+iη)`. The Wronskian is
`F′G − FG′ = 1`.

The order `l > −1/2` and the charge `η` are real. The radius is real and nonnegative, or
complex. Complex radii have a branch cut along the negative real axis, and the sign of a zero
imaginary part selects the side of the cut. At `z = 0` only `l = η = 0` is accepted, where the
result is `[0, 1, 1, 0]`.

The result is `[F, G, F′, G′]`. [`crate::coulomb_wave_with_options`] returns a
[`crate::CoulombWaveResult`] with error estimates and scaled forms for values outside the
range of the scalar type.

For positive real `z` the outgoing asymptotic expansion (DLMF 33.11.1) is used where
`(l² + η²)/z` is below about 59; it costs a few hundred work units. Elsewhere Steed's method
evaluates the continued fractions of DLMF 33.8 at a cost of about `z − η` units. With the
default budget of 65,536 units this covers `l, |η| ≤ 1300` at every radius from `2e4`; larger
orders or charges at radii from about `6e4` to `(l² + η²)/59` need a larger budget. Complex
radii combine the series at the origin, Taylor-series continuation of the equation, and the
expansion and continued fraction for `H± = G ± iF`.

# Examples

```
// F₀(0, z) = sin z and G₀(0, z) = cos z
let [f, g, f_prime, g_prime] = scioxide_special::coulomb_wave(0.0_f64, 0.0, 1.0_f64)?;
assert!((f - 1.0_f64.sin()).abs() < 1e-12);
assert!((g - 1.0_f64.cos()).abs() < 1e-12);
// Wronskian F′G − FG′ = 1
assert!((f_prime * g - f * g_prime - 1.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range of real arguments the error is within 64 ulps for `f64` and `f32`,
with absolute error bounds near zeros of the waves. Complex radii have no uniform ulp bound;
they are tested for orders up to 50, `|η| ≤ 100` and `1e-3 ≤ |z| ≤ 1e4`. The error estimate
of the detailed result includes recurrence, continuation and normalization error. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `l ≤ −1/2`, or if `z` is real and negative.
- `DomainError` (`Pole`) at `z = 0` for `l ≥ 0`, except `l = η = 0`.
- `DomainError` (`OutsideDomain`) at `z = 0` for `l < 0`.
- `NumericalInstability` (`Overflow`) if a component exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget.

# References

- NIST DLMF, Chapter 33. <https://dlmf.nist.gov/33>
- A. R. Barnett, Comput. Phys. Commun. 21 (1981) 297–314 (Steed's method)."###)]

    pub coulomb_wave [T: NumericScalar] (l: T::Real,
        eta: T::Real,
        z: T) -> [T; 4]
    where [];
    options options: CoulombOptions;
    detailed coulomb_wave_with_options -> CoulombWaveResult<T>;
    value {
        let (l, eta, z) = validate::<T>(l, eta, z)?;
        let mut w = Work::new(CoulombOptions::default().iteration_budget().get());
        let estimates = wave_estimates(l, eta, z, &mut w);
        let value = |estimate| {
            crate::complex_support::report_plain::<T>(estimate, &w, algorithm_component!("coulomb"))?
                .value()
        };
        match estimates {
            Ok([f, g, f_prime, g_prime]) => Ok([
                value(Ok(f))?,
                value(Ok(g))?,
                value(Ok(f_prime))?,
                value(Ok(g_prime))?,
            ]),
            // A failed kernel projects to its iteration-limit or numerical error.
            Err(e) => value(Err(e)).map(|v| [v; 4]),
        }
    }
    report {
        wave_run(l, eta, z, options, None)
    }
    scoped pub coulomb_wave_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            wave_run(l, eta, z, options, Some(context))
        })
    }
}

use CoulombComponent as Component;
fn origin(
    kind: Component,
    l: f64,
    eta: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let norm = kernel::normalization(l, eta, w)?;
    let log = norm.log;
    let (v, d) = match kind {
        Component::G => (1.0, 0.0),
        Component::GPrime => (0.0, 0.0),
        Component::F => (0.0, if l == 0.0 { 1.0 } else { 0.0 }),
        Component::FPrime => (if l == 0.0 { 1.0 } else { 0.0 }, 0.0),
        Component::SphericalF => (
            if l == 0.0 { 1.0 } else { 0.0 },
            if l == 0.0 {
                eta
            } else if l == 1.0 {
                1.0
            } else {
                0.0
            },
        ),
    };
    let log = if matches!(kind, Component::G | Component::GPrime) {
        -log
    } else {
        log
    };
    Ok(kernel::State {
        y: real(v),
        d: real(d),
        log: log.hi,
        log_lo: log.lo,
        error: (norm.error + 8.0 * f64::EPSILON) * (v.abs() + d.abs()),
    }
    .estimate(false, algorithm_component!("coulomb_origin_limit")))
}
fn component_run<T: NumericScalar>(
    kind: Component,
    l: T::Real,
    eta: T::Real,
    z: T,
    options: CoulombOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<crate::complex_support::ScaledEvaluation<T>, SciOxideError> {
    let (l, eta, point) = parameters::<T>(
        Request::Component(kind),
        l,
        eta,
        Some(z),
        1,
        scioxide_core::argument!("l"),
    )?;
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    w.check()?;
    fn component_estimate(
        kind: Component,
        l: f64,
        eta: f64,
        point: C,
        w: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        if point == ZERO {
            return origin(kind, l, eta, w);
        }
        let mut s = if matches!(
            kind,
            Component::F | Component::FPrime | Component::SphericalF
        ) {
            kernel::regular(l, eta, point, w)?
        } else {
            kernel::waves(l, eta, point, w)?.1
        };
        if matches!(kind, Component::SphericalF) {
            s.d = (s.d - s.y / point) / point;
            s.y /= point;
            s.error *= (1.0 + 1.0 / point.norm()) / point.norm();
        }
        Ok(s.estimate(
            matches!(kind, Component::FPrime | Component::GPrime),
            algorithm_component!("scaled_coulomb"),
        ))
    }
    let value = component_estimate(kind, l, eta, point, &mut w);
    crate::complex_support::report(
        value,
        !matches!(kind, Component::FPrime | Component::GPrime)
            && !(point == ZERO
                && (l < 0.0
                    || (matches!(kind, Component::G) && eta != 0.0)
                    || (matches!(kind, Component::SphericalF) && l > 0.0 && l < 1.0))),
        &w,
        algorithm_component!("coulomb"),
    )
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the regular Coulomb wave function `F_l(η,z)`.

`F_l` is the solution of the radial Coulomb equation that behaves as `C_l(η) z^(l+1)` as
`z → 0`. [`crate::coulomb_wave`] gives the equation, the normalization, the branch cut and the
methods, and computes `F`, `G` and both derivatives in one evaluation. The order `l > −1/2` and
the charge `η` are real. At `z = 0` the result is `0` for every accepted order.

# Examples

```
// F₀(0, z) = sin z
let f = scioxide_special::coulomb_f(0.0_f64, 0.0, 1.0_f64)?;
assert!((f - 1.0_f64.sin()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range of real arguments the error is within 64 ulps for `f64` and `f32`,
with absolute error bounds near zeros. Complex radii have no uniform ulp bound. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `l ≤ −1/2`, or if `z` is real and negative.
- `NumericalInstability` (`Overflow`) if `|F_l(η,z)|` exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget.

# References

NIST DLMF, §33.2. <https://dlmf.nist.gov/33.2>"###)]

    pub coulomb_f [T: NumericScalar] (l: T::Real, eta: T::Real, z: T) -> T
    where [];
    options options: CoulombOptions;
    detailed coulomb_f_with_options -> SpecialFunctionResult<T>;
    value {
        let options: CoulombOptions = CoulombOptions::default();
        component_run(Component::F, l, eta, z, options, None)?.value()
    }
    report {
        component_run(Component::F, l, eta, z, options, None)?.publish()
    }
    scoped pub coulomb_f_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            component_run(Component::F, l, eta, z, options, Some(context))?.publish()
        })
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the irregular Coulomb wave function `G_l(η,z)`.

`G_l` is the solution of the radial Coulomb equation with `G_l ~ cos θ` for large real `z`.
[`crate::coulomb_wave`] gives the equation, the phase `θ`, the branch cut and the methods. The
order `l > −1/2` and the charge `η` are real. At `z = 0` only `l = 0` is accepted, where
`G₀(η, 0) = 1/C₀(η)` with `C₀(η)` from [`crate::coulomb_normalization`].

# Examples

```
// G₀(0, z) = cos z
let g = scioxide_special::coulomb_g(0.0_f64, 0.0, 1.0)?;
assert!((g - 1.0_f64.cos()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range of real arguments the error is within 64 ulps for `f64` and `f32`,
with absolute error bounds near zeros. Complex radii have no uniform ulp bound. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `l ≤ −1/2`, or if `z` is real and negative.
- `DomainError` (`Pole`) at `z = 0` for `l > 0`.
- `DomainError` (`OutsideDomain`) at `z = 0` for `l < 0`.
- `NumericalInstability` (`Overflow`) if `|G_l(η,z)|` exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget.

# References

NIST DLMF, §33.2. <https://dlmf.nist.gov/33.2>"###)]

    pub coulomb_g [T: NumericScalar] (l: T::Real, eta: T::Real, z: T) -> T
    where [];
    options options: CoulombOptions;
    detailed coulomb_g_with_options -> SpecialFunctionResult<T>;
    value {
        let options: CoulombOptions = CoulombOptions::default();
        component_run(Component::G, l, eta, z, options, None)?.value()
    }
    report {
        component_run(Component::G, l, eta, z, options, None)?.publish()
    }
    scoped pub coulomb_g_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            component_run(Component::G, l, eta, z, options, Some(context))?.publish()
        })
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the derivative `dF_l(η,z)/dz` of the regular Coulomb wave function.

The derivative is taken with respect to `z` at fixed `l` and `η`. [`crate::coulomb_wave`] gives
the definition of `F_l`, the branch cut and the methods. The order `l > −1/2` and the charge `η`
are real. At `z = 0` the result is `C₀(η)` for `l = 0` and `0` for `l > 0`.

# Examples

```
// F₀(0, z) = sin z, so F₀′(0, z) = cos z
let f_prime = scioxide_special::coulomb_f_prime(0.0_f64, 0.0, 1.0)?;
assert!((f_prime - 1.0_f64.cos()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range of real arguments the error is within 64 ulps for `f64` and `f32`,
with absolute error bounds near zeros. Complex radii have no uniform ulp bound. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `l ≤ −1/2`, if `z` is real and negative, or if `z = 0`
  and `l < 0`.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget.

# References

NIST DLMF, §33.2. <https://dlmf.nist.gov/33.2>"###)]

    pub coulomb_f_prime [T: NumericScalar] (l: T::Real, eta: T::Real, z: T) -> T
    where [];
    options options: CoulombOptions;
    detailed coulomb_f_prime_with_options -> SpecialFunctionResult<T>;
    value {
        let options: CoulombOptions = CoulombOptions::default();
        component_run(Component::FPrime, l, eta, z, options, None)?.value()
    }
    report {
        component_run(Component::FPrime, l, eta, z, options, None)?.publish()
    }
    scoped pub coulomb_f_prime_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            component_run(Component::FPrime, l, eta, z, options, Some(context))?.publish()
        })
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the derivative `dG_l(η,z)/dz` of the irregular Coulomb wave function.

The derivative is taken with respect to `z` at fixed `l` and `η`. [`crate::coulomb_wave`] gives
the definition of `G_l`, the branch cut and the methods. The order `l > −1/2` and the charge `η`
are real. At `z = 0` only `l = η = 0` is accepted, where the result is `0`.

# Examples

```
// G₀(0, z) = cos z, so G₀′(0, z) = −sin z
let g_prime = scioxide_special::coulomb_g_prime(0.0_f64, 0.0, 1.0)?;
assert!((g_prime + 1.0_f64.sin()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range of real arguments the error is within 64 ulps for `f64` and `f32`,
with absolute error bounds near zeros. Complex radii have no uniform ulp bound. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `l ≤ −1/2`, or if `z` is real and negative.
- `DomainError` (`Pole`) at `z = 0` for `l ≥ 0`, except `l = η = 0`.
- `DomainError` (`OutsideDomain`) at `z = 0` for `l < 0`.
- `NumericalInstability` (`Overflow`) if the derivative exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget.

# References

NIST DLMF, §33.2. <https://dlmf.nist.gov/33.2>"###)]

    pub coulomb_g_prime [T: NumericScalar] (l: T::Real, eta: T::Real, z: T) -> T
    where [];
    options options: CoulombOptions;
    detailed coulomb_g_prime_with_options -> SpecialFunctionResult<T>;
    value {
        let options: CoulombOptions = CoulombOptions::default();
        component_run(Component::GPrime, l, eta, z, options, None)?.value()
    }
    report {
        component_run(Component::GPrime, l, eta, z, options, None)?.publish()
    }
    scoped pub coulomb_g_prime_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            component_run(Component::GPrime, l, eta, z, options, Some(context))?.publish()
        })
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the spherical regular Coulomb wave `F_l(η,z)/z`.

The quotient is the regular solution of the Coulomb problem in spherical form, and it stays
finite at the origin. [`crate::coulomb_wave`] gives the definition of `F_l`, the branch cut and
the methods. The order `l > −1/2` and the charge `η` are real. At `z = 0` the result is `C₀(η)`
for `l = 0` and `0` for `l > 0`.

# Examples

```
// F₀(0, z)/z = sin(z)/z
let f = scioxide_special::coulomb_spherical_f(0.0_f64, 0.0, 1.0)?;
assert!((f - 1.0_f64.sin()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"On the tested range of real arguments the error is within 64 ulps for `f64` and `f32`,
with absolute error bounds near zeros. Complex radii have no uniform ulp bound. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `l ≤ −1/2`, if `z` is real and negative, or if `z = 0`
  and `l < 0`.
- `NumericalInstability` (`Overflow`) if the result exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget.

# References

NIST DLMF, §33.2. <https://dlmf.nist.gov/33.2>"###)]

    pub coulomb_spherical_f [T: NumericScalar] (l: T::Real,
        eta: T::Real,
        z: T) -> T
    where [];
    options options: CoulombOptions;
    detailed coulomb_spherical_f_with_options -> SpecialFunctionResult<T>;
    value {
        let options: CoulombOptions = CoulombOptions::default();
        component_run(Component::SphericalF, l, eta, z, options, None)?.value()
    }
    report {
        component_run(Component::SphericalF, l, eta, z, options, None)?.publish()
    }
    scoped pub coulomb_spherical_f_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            component_run(Component::SphericalF, l, eta, z, options, Some(context))?.publish()
        })
    }
}

fn normalization_run<R: RealScalar + NumericScalar<Real = R>>(
    l: R,
    eta: R,
    options: CoulombOptions,
) -> Result<crate::complex_support::ScaledEvaluation<R>, SciOxideError> {
    let (l, eta, _) = parameters::<R>(
        Request::Normalization,
        l,
        eta,
        None,
        1,
        scioxide_core::argument!("l"),
    )?;
    let mut w = Work::new(options.iteration_budget().get());
    w.check()?;
    let e = kernel::normalization(l, eta, &mut w).map(kernel::Normalization::estimate);
    crate::complex_support::report(e, false, &w, algorithm_component!("coulomb"))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Coulomb normalization constant `C_l(η)`.

```text
C_l(η) = 2^l e^(−πη/2) |Γ(l+1+iη)| / Γ(2l+2),
```

for real order `l > −1` and real charge `η`. `C_l(η)` is positive, and `F_l(η,z) ~ C_l(η) z^(l+1)`
as `z → 0`. For large positive `η` it underflows; the detailed form
[`crate::coulomb_normalization_with_options`] then keeps its scaled and logarithmic forms. The
logarithm of `|Γ(l+1+iη)|` is evaluated in double-word arithmetic.

# Examples

```
// C₀(0) = 1 and C₁(0) = 2 Γ(2)/Γ(4) = 1/3
let c0 = scioxide_special::coulomb_normalization(0.0_f64, 0.0)?;
let c1 = scioxide_special::coulomb_normalization(1.0_f64, 0.0)?;
assert!((c0 - 1.0).abs() < 1e-12);
assert!((c1 - 1.0 / 3.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. The detailed result reports an error estimate; tests cover
`|η|` up to `1e4`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `l ≤ −1`.

# References

NIST DLMF, §33.2. <https://dlmf.nist.gov/33.2>"###)]

    pub coulomb_normalization [R: RealScalar + NumericScalar<Real = R>] (l: R,
        eta: R) -> R
    where [];
    options options: CoulombOptions;
    detailed coulomb_normalization_with_options -> SpecialFunctionResult<R>;
    evaluate |options, context| {
        normalization_run(l, eta, options)
    }

}

pub(super) fn hydrogenic_run<T: NumericScalar>(
    n: u32,
    l: u32,
    charge: T::Real,
    r: T,
    options: CoulombOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<crate::complex_support::ScaledEvaluation<T>, SciOxideError> {
    let charge = crate::support::widen_numeric(charge, "Coulomb parameter charge must be finite")
        .map(|v| v.re)?;
    let r = widen(r, "Coulomb parameter r must be finite")?;
    if n == 0 {
        return Err(crate::support::COULOMB_ERRORS.outside(scioxide_core::argument!("n")));
    }
    if l >= n {
        return Err(crate::support::COULOMB_ERRORS.outside(scioxide_core::argument!("l")));
    }
    if charge <= 0.0 {
        return Err(crate::support::COULOMB_ERRORS.outside(scioxide_core::argument!("charge")));
    }
    if is_real::<T>() && r.re < 0.0 {
        return Err(crate::support::COULOMB_ERRORS.outside(scioxide_core::argument!("r")));
    }
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    w.check()?;
    fn hydrogenic_estimate(
        n: u32,
        l: u32,
        charge: f64,
        r: C,
        w: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        let nf = f64::from(n);
        let lf = f64::from(l);
        let m = n - l - 1;
        let rate = 2.0 * charge / nf;
        let (gm, _) =
            crate::gamma_beta::gamma::complex_log_gamma(real(f64::from(m) + 1.0), &mut || {
                w.try_step()
            })?;
        let (gn, _) =
            crate::gamma_beta::gamma::complex_log_gamma(real(nf + lf + 1.0), &mut || w.try_step())?;
        // U(-m,b,x)=(-1)^m m! L_m^(b-1)(x); its terminating kernel owns the recurrence.
        let mut e = crate::hypergeometric_family::hypergeometric_kernel::hyperu(
            Jet::from(-f64::from(m)),
            Jet::from(2.0 * lf + 2.0),
            Jet {
                v: real(rate) * r,
                d: real(rate),
            },
            w,
        )?;
        let log_norm = 1.5 * rate.ln() - 0.5 * (2.0 * nf).ln() - 0.5 * (gm.re + gn.re);
        if m % 2 == 1 {
            e.y = -e.y;
        }
        if r == ZERO {
            if l == 1 {
                e.y.d = e.y.v * real(rate);
                e.y.v = ZERO;
            } else if l > 1 {
                e.y = Jet::from(0.0);
            } else {
                let value = e.y.v;
                e.y.d -= value * real(rate / 2.0);
            }
            e.scale = e.scale + Jet::from(log_norm);
        } else {
            let logarithm = (real(rate) * r).ln();
            e.scale = e.scale
                + Jet {
                    v: real(log_norm) + real(lf) * logarithm - real(rate / 2.0) * r,
                    d: real(lf) / r - real(rate / 2.0),
                };
        }
        e.method = algorithm_component!("hydrogenic_terminating_U");
        Ok(e)
    }
    let e = hydrogenic_estimate(n, l, charge, r, &mut w);
    crate::complex_support::report(e, true, &w, algorithm_component!("hydrogenic"))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the radial function `R_nl(Z,r)` of a hydrogen-like bound state.

```text
R_nl(Z,r) = (2Z/n)^(3/2) √((n−l−1)! / (2n (n+l)!)) e^(−ρ/2) ρ^l L_(n−l−1)^(2l+1)(ρ),   ρ = 2Zr/n,
```

where `n ≥ 1` is the principal quantum number, `0 ≤ l < n` the orbital quantum number, `Z > 0`
the nuclear charge, `r` the radius in Bohr radii and `L_k^(α)` the generalized Laguerre
polynomial (DLMF §18.3). The normalization is `∫₀^∞ R_nl(Z,r)² r² dr = 1`, without an angular
factor. The radius is real and nonnegative, or complex; `R_nl` is entire in `r`. The Laguerre
polynomial is evaluated as the terminating Kummer function `U(−(n−l−1), 2l+2, ρ)`.

# Examples

```
// R₁₀(Z, r) = 2 Z^(3/2) e^(−Zr)
let at_origin = scioxide_special::hydrogenic_radial(1, 0, 1.0_f64, 0.0_f64)?;
let at_one = scioxide_special::hydrogenic_radial(1, 0, 1.0_f64, 1.0_f64)?;
assert!((at_origin - 2.0).abs() < 1e-12);
assert!((at_one - 2.0 * (-1.0_f64).exp()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. The detailed result reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `n = 0`, `l ≥ n`, `Z ≤ 0`, or `r` is real and negative.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget."###)]

    pub hydrogenic_radial [T: NumericScalar] (n: u32,
        l: u32,
        charge: T::Real,
        r: T) -> T
    where [];
    options options: CoulombOptions;
    detailed hydrogenic_radial_with_options -> SpecialFunctionResult<T>;
    value {
        let options: CoulombOptions = CoulombOptions::default();

        hydrogenic_run(n, l, charge, r, options, None)?.value()
    }
    report {
        hydrogenic_run(n, l, charge, r, options, None)?.publish()
    }
    scoped pub hydrogenic_radial_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            hydrogenic_run(n, l, charge, r, options, Some(context))?.publish()
        })
    }
}
